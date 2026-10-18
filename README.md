# HILabs — Clinical Term Mapping (ICD-10-CM + SNOMED CT)

Map a free-text clinical entity (+ note context) to **one ICD-10-CM code** and **one
SNOMED CT concept id**, with calibrated confidence.

## Results

| split | n questions | ICD-10-CM | SNOMED CT | overall |
|---|---|---|---|---|
| dev  (config A, dev-selected) | 780 | **0.573** (208/363) | **0.667** (278/417) | **0.620** |
| test (config A, single-pass) | 1,025 | 0.532 (249/468) | 0.603 (336/557) | 0.571 |
| test (config B, permute n=8) | 1,025 | **0.564** (264/468) | **0.612** (341/557) | **0.590** |

Diagnostics (test, config B): shortlist recall 0.838 / 0.874; acc|present 0.673 / 0.700;
ICD three-character-block partial credit 0.748; unseen-gold accuracy 0.507 / 0.564.

**Published models**
- Chooser (Kev-4B LoRA + pointer head): https://huggingface.co/d3let3/kev-4b-clinical-mapping
- Pairwise judge (UltraBERT-150M): https://huggingface.co/d3let3/ultrabert-clinical-judge


## Architecture

```
entity + chunk context
  └─ candidate sources: exact · memory · lexical · dense (ub5: gte-base + BioLORD +
       UltraBERT-Embed) · fuzzy · crosswalk          → source fusion
       └─ encoder judges: UltraBERT-150M pairwise (fine-tuned) + Qwen3-Reranker-8B
            shortlist: rank = w·z(judge) + (1−w)·z(fused)   [ICD K=10; SNOMED K=30]
       └─ Kev-4B chooser (Qwen3.5-4B-Base + LoRA + pointer head, init jaredpalmer/kev-4b)
            → calibrated probabilities over the shortlist
       └─ permutation ensemble: /v1/systemone/permute n=8, mean probability → argmax
            + stability and margin for abstention/routing
```

The two-stage design is deliberate: retrieval is near its ceiling (0.82/0.89 dev recall at
small K), while disambiguation dominates the remaining error. A small calibrated decision
model is more accurate and far cheaper than a generative LLM for this constrained choice.

## Repository layout

```
predict.py                 single-command inference (auto | kev | engine modes)
src/term_map/              mapping engine (normalisation, terminology, candidates,
                           fusion, decision policy, dense encoders, judges, pipeline)
scripts/                   00–03 data prep/lexicon/embeddings/indexes · 04 engine CLI
                           05 eval · 06–07 policy train/tune · 08 candidate export
                           09 chooser eval · 09b permutation-ensemble eval
                           10–12 UltraBERT judge training and scoring
                           14 shortlist builder · 16 Qwen3 reranker scoring
configs/default.yaml       engine configuration
data/                      test/dev inputs + final shortlists used for the reported numbers
results/                   per-question predictions, analysis JSON, summary
predictions/               submitted predictions (schema: chunk_id, clinical_entity,
                           icd10cm_code, snomed_code)
reports/                   LaTeX reports (project + architecture) and compiled PDFs
demo/                      storyboard, hard cases, live inspection script
tests/                     unit tests for the engine
```

## Setup

Python 3.10+. Tested with torch 2.14 + transformers 5.20 on CUDA.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dense,gpu]"
cp .env.example .env          # fill in what you need (nothing is required for mode=auto)
```

Environment variables (see `.env.example`): `TERMMAP_CONFIG`, `KEV_ENDPOINT`, `KEV_MODEL`,
optional `LLM_*`, `HF_TOKEN`. Never commit real keys.

The mapping engine additionally needs the processed terminology, splits, lexicon, policies
and embedding/index artifacts (`data/processed/`, `artifacts/`); these are produced by the
data scripts (`scripts/00`–`03`, `06`–`07`) and the judge-training scripts (`scripts/10`–`12`),
and are not checked in (size).

## Run

### 1. Reproduce the submitted predictions (single command)

```bash
python predict.py --input data/test_input.csv --output predictions/test_predictions.csv
```

This joins the stored chooser picks (config B, permutation ensemble n=8) through the
shortlist metadata and fills rows without a chooser question from the deterministic engine.
Output schema: `chunk_id, clinical_entity, icd10cm_code, snomed_code` (577 rows, complete).

### 2. Live two-stage run (needs the Kev server and the fine-tuned checkpoint)

```bash
python -m kev.serve --run d3let3/kev-4b-clinical-mapping --port 8013 --device cuda
KEV_ENDPOINT=http://127.0.0.1:8013 \
python predict.py --input data/test_input.csv --output /tmp/live.csv --mode kev --nperm 8
```

The fine-tuned checkpoint (LoRA adapter + pointer head on Qwen3.5-4B-Base, ~135 MB) is
public on the Hub and loads directly by repo id. Verification: serving the Hub checkpoint on
the dev shortlist reproduces ICD 0.548 / SNOMED 0.647 single-pass (0.573 / 0.667 with the
8-permutation ensemble).

### 3. Deterministic engine only (fully offline, no chooser)

```bash
python predict.py --input data/test_input.csv --output /tmp/engine.csv --mode engine
```

Engine-only test accuracy is 0.440 / 0.391 (ICD / SNOMED); it exists as a fallback and as a
reproducible baseline, not as the headline system. It runs at ~4 rows/s on CPU.

### 4. Evaluate a shortlist

```bash
python scripts/09_eval_kev.py --jsonl data/dev_shortlist.jsonl \
    --url http://127.0.0.1:8013 --out /tmp/dev_single.csv
python scripts/09b_eval_permute.py --jsonl data/dev_shortlist.jsonl \
    --csv /tmp/dev_single.csv --url http://127.0.0.1:8013 --nperm 8 \
    --out /tmp/dev_permute8.csv
```

### 5. Build a shortlist from judge scores

```bash
python scripts/14_build_kev_shortlist.py --split dev --k-icd 10 --k-snomed 30 \
  --judge-scores-icd out/kev/dev_ub5_scores.jsonl --judge-weight-icd 0.2 \
  --judge-scores-snomed out/kev/dev_combined_scores.jsonl --judge-weight-snomed 0.6 \
  --out out/kev/dev_ub5_sl_hybrid.jsonl
```

## Demo

`demo/README.md` has the 2–3 minute storyboard and the recorded reference results for seven
hard cases (abbreviation, abbreviation collision, OCR artifact, specificity, convention,
sibling near-miss). Run it live:

```bash
export KEV_ENDPOINT=http://127.0.0.1:8013
python demo/inspect.py --nperm 8
```

Without a server, `python demo/inspect.py` prints the recorded picks, margins and stability
for the same rows.

## Reports

- `reports/report_project.pdf` — problem, system evolution, engineering challenges, all
  experiments (including what failed), results, error analysis, lessons, next steps.
- `reports/report_architecture.pdf` — full architecture: sources and fusion, judges and
  shortlists, Kev chooser, permutation ensembling, decision layer, training protocol,
  ablations, deployment characteristics.
- `reports/deck_final.pdf` — 8-slide presentation deck: problem, architecture, both stages,
  results, error analysis + demo cases, lessons and next steps.

## Data and external resources

Terminology: ICD-10-CM FY2026 descriptions, SNOMED CT descriptions (2026-09-01 release),
official SNOMED→ICD crosswalk. Models: `thenlper/gte-base`, `FremyCompany/BioLORD-2023`,
`dmis-lab/UltraBERT-Embed-150M`, fine-tuned `dmis-lab/UltraBERT-150M` judge,
`Qwen/Qwen3-Reranker-8B`, and the Kev-4B decision model
([jaredpalmer/kev](https://github.com/jaredpalmer/kev)) fine-tuned on
`Qwen/Qwen3.5-4B-Base`.

## Limitations

- The largest error group is dataset-specific specificity convention (e.g. gold `E78.0` vs
  billable child `E78.00`), which no amount of generic reasoning fixes; it needs
  convention-aware training or adjudication.
- ICD shortlist recall on the test distribution (0.838 at K=20) is the biggest accuracy cap.
- SNOMED partial credit (IS-A distance) is not reported; exact match only.
