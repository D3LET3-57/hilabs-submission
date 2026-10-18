---
base_model: Qwen/Qwen3.5-4B-Base
library_name: peft
license: apache-2.0
tags:
  - clinical-coding
  - icd-10-cm
  - snomed-ct
  - decision-model
  - kev
  - term-mapping
---

# kev-4b-clinical-mapping

A fine-tuned **Kev-4B decision model** for clinical term mapping: given a clinical entity and
its note context plus a shortlist of candidates, it returns calibrated probabilities over the
candidates (no text generation). It is the second stage of the HILabs ICD-10-CM / SNOMED CT
mapping system.

- Base: `Qwen/Qwen3.5-4B-Base`, initialised from `jaredpalmer/kev-4b` (Kev 1.0: LoRA adapter
  + pointer head), then fine-tuned on task data.
- Runtime: [jaredpalmer/kev](https://github.com/jaredpalmer/kev) (`kev.serve`,
  `POST /v1/systemone`), Apache-2.0.
- Training: 1,776 entity/context examples; LoRA r=16, alpha=32; 2 epochs; lr 2e-5; bf16;
  shortlists built with the exact inference recipe.

## Usage

```bash
git clone https://github.com/jaredpalmer/kev && cd kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run <this-repo-id> --port 8013 --device cuda
```

```bash
curl -s localhost:8013/v1/systemone/permute -H 'content-type: application/json' -d '{
  "request": {"state": {"entity": "t2dm poorly controlled",
                        "context": "t2dm poorly controlled: transitioned off insulin"},
               "model": "kev-latest",
               "questions": {"icd_pick": {"type": "choice",
                 "instructions": "You are a certified medical coder. Choose the ICD-10-CM code that best matches the clinical concept in the patient context.",
                 "criteria": {"E11.9": "Type 2 diabetes mellitus without complications",
                              "E11.65": "Type 2 diabetes mellitus with hyperglycemia"}}}},
  "question": "icd_pick", "n_perm": 8, "seed": 0}'
```

Ensembling the calibrated probabilities over several option orders (`/permute`, n=8) is part
of the intended inference procedure.

## Results (internal splits, exact match)

| split | ICD-10-CM | SNOMED CT |
|---|---|---|
| dev  | 0.573 (208/363) | 0.667 (278/417) |
| test | 0.564 (264/468) | 0.612 (341/557) |

Accuracy given the gold is in the shortlist: 0.70 / 0.75 (dev), 0.67 / 0.70 (test).

## Limitations

- Trained for shortlist disambiguation, not open-vocabulary code generation; it expects a
  small candidate list with descriptions.
- The dataset's coding conventions (e.g. parent vs billable child) dominate the remaining
  errors; the model reproduces the conventions it saw in training.
- Trained on de-identified clinical notes; the adapter does not contain the raw notes, but
  review your own data-governance requirements before redistribution.
