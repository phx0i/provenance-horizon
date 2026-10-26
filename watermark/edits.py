"""Permitted edit channels: identity, light, DIPPER, GPT-4o, recursive, and the adaptive editor."""
import os
import re
from dataclasses import dataclass

import numpy as np

from watermark.hostlm import chat_completion

DEFAULT_DISCLAIMER = (
    "Disclaimer: This material is for informational purposes only and does not constitute "
    "investment advice."
)
PARAPHRASE_PROMPT = (
    "Paraphrase the following text, preserving its meaning but changing the wording substantially."
)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,2}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


class EditChannel:
    name = "abstract"

    def describe(self):
        return {"name": self.name}

    def apply(self, text, rng):
        raise NotImplementedError


class IdentityChannel(EditChannel):
    name = "none"

    def apply(self, text, rng):
        return text


class SpacyRedactor:
    """Replaces named entities of the given labels and e-mail, phone and SSN patterns with a mask."""

    def __init__(self, model="en_core_web_sm", mask="[REDACTED]", labels=("PERSON",)):
        import spacy

        self._model = model
        self.nlp = spacy.load(model, disable=["parser", "lemmatizer"])
        self.mask = mask
        self.labels = set(labels)

    def describe(self):
        return {"type": "spacy", "model": self._model, "mask": self.mask, "labels": sorted(self.labels)}

    def __call__(self, text):
        doc = self.nlp(text)
        out, last = [], 0
        for ent in doc.ents:
            if ent.label_ in self.labels:
                out.append(text[last: ent.start_char])
                out.append(self.mask)
                last = ent.end_char
        out.append(text[last:])
        text = "".join(out)
        for pat in (_EMAIL_RE, _PHONE_RE, _SSN_RE):
            text = pat.sub(self.mask, text)
        return text


@dataclass
class LightEditChannel(EditChannel):
    """PII redaction of the text, then the disclaimer prepended."""

    redactor: object
    disclaimer: str = DEFAULT_DISCLAIMER
    name: str = "light"

    def describe(self):
        rd = self.redactor.describe() if hasattr(self.redactor, "describe") else {"type": type(self.redactor).__name__}
        return {"name": self.name, "disclaimer": self.disclaimer, "redactor": rd}

    def apply(self, text, rng):
        return f"{self.disclaimer} {self.redactor(text)}"


def simple_sentence_split(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text.strip())
    return [p for p in parts if p]


class DipperChannel(EditChannel):
    """DIPPER paraphraser over three-sentence windows with the paraphrased prefix as context."""

    name = "dipper"

    def __init__(self, model_name="kalpeshk2011/dipper-paraphraser-xxl", tokenizer_name="google/t5-v1_1-xxl",
                 lex_diversity=60, order_diversity=60, sent_interval=3, device=None, dtype="bfloat16",
                 top_p=0.75, max_length=512, revision=None, sentence_splitter=None):
        if lex_diversity % 20 or order_diversity % 20:
            raise ValueError("DIPPER diversity codes must be multiples of 20")
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.torch = torch
        self._model_name = model_name
        self._tokenizer_name = tokenizer_name
        self._revision = revision
        self._dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        kwargs = {"torch_dtype": getattr(torch, dtype)} if dtype else {}
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, revision=revision, **kwargs)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.lex_code = int(100 - lex_diversity)
        self.order_code = int(100 - order_diversity)
        self.sent_interval = sent_interval
        self.top_p = top_p
        self.max_length = max_length
        self.split = sentence_splitter or simple_sentence_split

    def describe(self):
        return {"name": self.name, "model": self._model_name, "tokenizer": self._tokenizer_name,
                "revision": self._revision, "dtype": self._dtype, "lex": 100 - self.lex_code,
                "order": 100 - self.order_code, "sent_interval": self.sent_interval,
                "top_p": self.top_p, "max_length": self.max_length}

    def apply(self, text, rng):
        torch = self.torch
        text = " ".join(text.split())
        sents = self.split(text)
        prefix, output = "", ""
        seed = int(rng.integers(0, 2**31 - 1))
        for i in range(0, len(sents), self.sent_interval):
            window = " ".join(sents[i: i + self.sent_interval])
            inp = f"lexical = {self.lex_code}, order = {self.order_code}"
            if prefix:
                inp += f" {prefix}"
            inp += f" <sent> {window} </sent>"
            enc = self.tokenizer([inp], return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            torch.manual_seed(seed + i)
            with torch.inference_mode():
                gen = self.model.generate(**enc, do_sample=True, top_p=self.top_p, top_k=None,
                                          max_length=self.max_length)
            piece = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
            prefix = (prefix + " " + piece).strip()
            output = (output + " " + piece).strip()
        return output


class GPT4oChannel(EditChannel):
    """Paraphrase through the OpenAI chat-completions API."""

    name = "gpt4o"

    def __init__(self, model="gpt-4o-2024-08-06", temperature=1.0, prompt=PARAPHRASE_PROMPT, api_key=None,
                 base_url="https://api.openai.com/v1", max_retries=5):
        import requests

        self.requests = requests
        self.model = model
        self.temperature = temperature
        self.prompt = prompt
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the GPT-4o channel")
        self.base_url = base_url
        self.max_retries = max_retries

    def describe(self):
        return {"name": self.name, "model": self.model, "temperature": self.temperature,
                "prompt": self.prompt, "base_url": self.base_url}

    def apply(self, text, rng):
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": f"{self.prompt}\n\n{text}"}],
        }
        return chat_completion(self.requests, self.base_url, self.api_key, body, self.max_retries)


@dataclass
class ComposedChannel(EditChannel):
    stages: list
    name: str = "recursive"

    def describe(self):
        return {"name": self.name, "stages": [s.describe() for s in self.stages]}

    def apply(self, text, rng):
        for st in self.stages:
            text = st.apply(text, rng)
        return text


@dataclass
class AdaptiveChannel(EditChannel):
    """Draws candidate outputs from the base channel and keeps the one least similar to the source under the encoder."""

    base: EditChannel
    encoder: object
    candidates: int = 8
    name: str = "adaptive"

    def describe(self):
        return {"name": self.name, "base": self.base.describe(), "encoder": self.encoder.describe(),
                "candidates": self.candidates}

    def apply(self, text, rng):
        cands = [self.base.apply(text, rng) for _ in range(self.candidates)]
        emb = self.encoder.encode([text] + cands)
        sims = emb[1:] @ emb[0]
        return cands[int(np.argmin(sims))]


def build_edit_channel(name, cfg, lm=None, cache=None, encoder=None):
    """Instantiates a channel by name from the edits config block; cache shares loaded models."""
    cache = cache if cache is not None else {}
    block = cfg.get(name, {})
    if name == "none":
        return IdentityChannel()
    if name == "light":
        red = SpacyRedactor(block.get("spacy_model", "en_core_web_sm"), mask=block.get("mask", "[REDACTED]"),
                            labels=tuple(block.get("entity_labels", ["PERSON"])))
        return LightEditChannel(disclaimer=block.get("disclaimer", DEFAULT_DISCLAIMER), redactor=red)
    if name == "dipper":
        if "dipper" not in cache:
            d = cfg.get("dipper", {})
            cache["dipper"] = DipperChannel(model_name=d.get("model", "kalpeshk2011/dipper-paraphraser-xxl"),
                                            tokenizer_name=d.get("tokenizer", "google/t5-v1_1-xxl"),
                                            lex_diversity=int(d.get("lex", 60)),
                                            order_diversity=int(d.get("order", 60)),
                                            sent_interval=int(d.get("sent_interval", 3)),
                                            device=d.get("device"), dtype=d.get("dtype", "bfloat16"),
                                            top_p=float(d.get("top_p", 0.75)), max_length=int(d.get("max_length", 512)),
                                            revision=d.get("revision"))
        return cache["dipper"]
    if name == "gpt4o":
        return GPT4oChannel(model=block.get("model", "gpt-4o-2024-08-06"),
                            temperature=float(block.get("temperature", 1.0)))
    if name == "recursive":
        d = build_edit_channel("dipper", cfg, lm=lm, cache=cache)
        return ComposedChannel(stages=[d, d], name="recursive")
    if name == "adaptive":
        if encoder is None:
            raise ValueError("adaptive channel needs the retrieval encoder")
        base = build_edit_channel(block.get("base", "dipper"), cfg, lm=lm, cache=cache, encoder=encoder)
        return AdaptiveChannel(base=base, encoder=encoder, candidates=int(block.get("candidates", 8)))
    raise ValueError(f"unknown edit channel {name!r}")
