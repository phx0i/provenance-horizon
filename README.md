# provenance-horizon

Computes the provenance horizon and the green-count lengths that bracket it, watermarks host-model outputs with the green-list and Unigram schemes and detects them under permitted edit channels, estimates the per-token distortion and the edit contraction, and runs a split-conformal retrieval-log provenance test.

## Install

Python 3.11 or 3.12.

```bash
pip install -r requirements.txt
```

## Run

```bash
export HF_TOKEN=...          # meta-llama/Meta-Llama-3-8B and meta-llama/Meta-Llama-3-70B are gated on the Hugging Face Hub
export OPENAI_API_KEY=...    # GPT-4o paraphrase channel and GPT-4o host
python main.py theory
python main.py prepare-data --config configs/full.yaml --user-agent "Org Name contact@example.org"
python main.py lengths      --config configs/full.yaml
python main.py coverage     --config configs/full.yaml
python main.py coverage     --config configs/full.yaml --sweep
python main.py calibration  --config configs/full.yaml
python main.py coverage     --config configs/host_llama3_8b.yaml
python main.py coverage     --config configs/host_llama3_70b.yaml
python main.py coverage     --config configs/host_gpt4o.yaml
```

## Layout

```
main.py         entry point
configs/        settings
theory/         closed-form quantities
watermark/      host models, watermarks, edit channels, estimators
retrieval/      retrieval score, conformal calibration
data/           EDGAR acquisition, lengths, splits, strata
evaluation/     experiment protocol, config loading, tables
```

| File | Description |
|---|---|
| `main.py` | Subcommands `theory`, `prepare-data`, `lengths`, `coverage`, `calibration`, `all` |
| `configs/full.yaml` | Full-scale settings |
| `configs/host_llama3_8b.yaml`, `host_llama3_70b.yaml`, `host_gpt4o.yaml` | Host-generator sweep settings |
| `theory/horizon.py` | Horizon, budget-consistent bias, Hoeffding, Gaussian and exact binomial lengths |
| `watermark/hostlm.py` | Local host model with seeded sampling and per-step KL; chat-completions host |
| `watermark/greenlist.py` | Green-list and Unigram watermarks, z-test |
| `watermark/edits.py` | Identity, light, DIPPER, GPT-4o, recursive and adaptive channels |
| `watermark/estimation.py` | Estimators of gamma and eta |
| `retrieval/index.py` | Embedding, n-gram, exact n-gram, LCS and hybrid similarities over the log |
| `retrieval/conformal.py` | Split-conformal and group-conditional thresholds, Beta spread, log commitment |
| `data/edgar.py` | EDGAR download and section extraction |
| `data/lengths.py` | Sentence segmentation, token counts, unit fractions |
| `data/splits.py` | Filing-level splits with null-only records, third-party windows, host prompts |
| `data/strata.py` | Boilerplate stratum |
| `evaluation/protocol.py` | Context construction, per-length log and calibration, per-seed evaluation, ablation, hard negatives, adaptive editor |
| `evaluation/config.py` | YAML loading |
| `evaluation/report.py` | Markdown tables |

## Configuration

| Key | Values |
|---|---|
| `alpha` | `0.01` (false-positive bound) |
| `beta` | `0.05` (one minus coverage floor) |
| `gamma_budget` | `0.02` (per-token distortion budget for the horizon) |
| `host.model` | `facebook/opt-6.7b` (host generator) or alternatives in `host_*.yaml` |
| `watermarks.*.scheme` | `greenlist` (context width 1) or `unigram` (context-free) |
| `watermarks.*.bias` | `2.0` (logit bias strength) |
| `watermarks.*.green_fraction` | `0.5` (green partition fraction) |
| `edits.dipper.lex` | `60` (DIPPER lexical diversity) |
| `edits.dipper.order` | `60` (DIPPER order diversity) |
| `edits.gpt4o.temperature` | `1.0` (GPT-4o paraphrase temperature) |
| `retrieval.index.score` | `hybrid` (scoring function) or `embed`, `ngram`, `exact_ngram`, `lcs` |
| `retrieval.index.embed_weight` | `0.7` (embedding weight in hybrid score) |
| `retrieval.index.ngram_weight` | `0.3` (n-gram weight in hybrid score) |
| `splits.calibration_size` | `2000` (conformal calibration set size) |
| `splits.log_size` | `5000` (output log size) |
| `n_positive` | `500` (positive examples per condition per seed) |
| `n_negative` | `500` (negative examples per condition per seed) |
| `seeds` | `[0, 1, 2, 3, 4]` (evaluation seeds) |
