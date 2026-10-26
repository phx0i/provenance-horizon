"""Host generators: an explicit sampling loop over local logits, and a chat-completions host for text-only runs."""
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GenerationResult:
    ids: list
    per_step_kl: list = field(default_factory=list)


def softmax(logits):
    z = logits - np.max(logits)
    p = np.exp(z)
    return p / p.sum()


def kl_divergence(p, q):
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


def chat_completion(session, base_url, api_key, body, max_retries):
    """Posts a chat-completions request, retrying failed attempts, and returns the stripped message content."""
    from requests import RequestException

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    err = ""
    for attempt in range(max_retries):
        try:
            r = session.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(body), timeout=120)
        except RequestException as e:
            err = repr(e)
        else:
            if r.status_code == 200:
                try:
                    content = r.json()["choices"][0]["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    raise RuntimeError(f"malformed chat completion response: {r.text[:200]}") from e
                if content is None:
                    raise RuntimeError(f"chat completion returned no message content: {r.text[:200]}")
                return content.strip()
            err = r.text[:200]
        time.sleep(2 ** attempt)
    raise RuntimeError(f"chat completion failed after {max_retries} attempts: {err}")


def top_p_sample(probs, top_p, rng):
    """Nucleus sampling over the smallest set of tokens with cumulative mass at least top_p."""
    if top_p >= 1.0:
        return int(rng.choice(len(probs), p=probs))
    order = np.argsort(-probs)
    sorted_p = probs[order]
    cum = np.cumsum(sorted_p)
    cutoff = int(np.searchsorted(cum, top_p)) + 1
    keep = order[:cutoff]
    p = probs[keep] / probs[keep].sum()
    return int(rng.choice(keep, p=p))


class HostLM:
    """Abstract host model: tokenisation and seeded sampling; subclasses supply the generation hooks."""

    vocab_size = 0
    name = "abstract"

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def generate(self, prompt_ids, n_tokens, seed, temperature=1.0, top_p=0.95, processor=None, record_kl=False):
        """Sample exactly n_tokens tokens; the processor biases logits before sampling."""
        rng = np.random.default_rng(seed)
        ids = list(prompt_ids)
        out = GenerationResult(ids=[])
        step_fn = self._begin_generation(ids)
        prev = ids[-1] if ids else None
        for _ in range(n_tokens):
            logits = step_fn() / temperature
            logits = self._suppress_eos(logits)
            biased = processor(logits, prev) if processor is not None else logits
            if record_kl:
                out.per_step_kl.append(kl_divergence(softmax(biased), softmax(logits)))
            tok = top_p_sample(softmax(biased), top_p, rng)
            out.ids.append(tok)
            self._advance(tok)
            prev = tok
        self._end_generation()
        return out

    def _suppress_eos(self, logits):
        return logits


class HFCausalLM(HostLM):
    """Causal language model loaded by name or path, optionally in 8-bit quantisation."""

    def __init__(self, model_name, device=None, dtype=None, revision=None, quantization=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        kwargs = {"torch_dtype": getattr(torch, dtype)} if dtype else {}
        if quantization == "int8":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        elif quantization is not None:
            raise ValueError(f"unknown quantization {quantization!r}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, **kwargs)
        self.model.eval()
        if quantization is None:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        else:
            self.device = self.model.device
        self.vocab_size = int(self.model.get_output_embeddings().weight.shape[0])
        self.eos_id = self.tokenizer.eos_token_id
        self._cache = None
        self._pending = []

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    @property
    def _start_id(self):
        return self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.eos_id

    def _with_start(self, ids):
        ids = list(ids)
        return ids if ids[:1] == [self._start_id] else [self._start_id] + ids

    def _begin_generation(self, ids):
        torch = self.torch
        self._cache = None
        self._pending = self._with_start(ids)

        def step():
            with torch.inference_mode():
                x = torch.tensor([self._pending], device=self.device)
                out = self.model(x, past_key_values=self._cache, use_cache=True)
                self._cache = out.past_key_values
                self._pending = []
                return out.logits[0, -1].float().cpu().numpy()

        return step

    def _advance(self, tok):
        self._pending.append(tok)

    def _end_generation(self):
        self._cache = None
        self._pending = []

    def _suppress_eos(self, logits):
        if self.eos_id is not None:
            logits = logits.copy()
            logits[self.eos_id] = -np.inf
        return logits


class OpenAIChatLM(HostLM):
    """Chat-completions host: text generation only; windows and prompts are tokenised by a local tokenizer."""

    def __init__(self, model="gpt-4o-2024-08-06", tokenizer="meta-llama/Meta-Llama-3-8B", tokenizer_revision=None,
                 api_key=None, base_url="https://api.openai.com/v1", max_retries=5):
        import requests
        from transformers import AutoTokenizer

        self.requests = requests
        self.name = model
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision=tokenizer_revision)
        self.vocab_size = len(self.tokenizer)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the chat-completions host")
        self.base_url = base_url
        self.max_retries = max_retries

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def generate(self, prompt_ids, n_tokens, seed, temperature=1.0, top_p=0.95, processor=None, record_kl=False):
        """Requests a completion of at most n_tokens tokens; logit processing and per-step KL are unavailable."""
        if processor is not None or record_kl:
            raise ValueError("the chat-completions host exposes no logits: watermarks and per-step KL are unavailable")
        body = {"model": self.model, "temperature": temperature, "top_p": top_p, "max_tokens": int(n_tokens),
                "seed": int(seed), "messages": [{"role": "user", "content": self.decode(prompt_ids)}]}
        text = chat_completion(self.requests, self.base_url, self.api_key, body, self.max_retries)
        return GenerationResult(ids=self.encode(text))


def load_host_lm(cfg):
    kind = cfg.get("type", "hf")
    if kind == "hf":
        return HFCausalLM(cfg["model"], device=cfg.get("device"), dtype=cfg.get("dtype"), revision=cfg.get("revision"),
                          quantization=cfg.get("quantization"))
    if kind == "openai":
        return OpenAIChatLM(model=cfg.get("model", "gpt-4o-2024-08-06"),
                            tokenizer=cfg.get("tokenizer", "meta-llama/Meta-Llama-3-8B"),
                            tokenizer_revision=cfg.get("tokenizer_revision"))
    raise ValueError(f"unknown host LM type {kind!r}")
