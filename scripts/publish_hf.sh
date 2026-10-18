#!/bin/bash
# Publish the trained artifacts to the Hugging Face Hub.
# Requires: export HF_TOKEN=hf_...   (a *write* token from hf.co/settings/tokens)
# Optional overrides: HF_USER, CHOOSER_REPO, JUDGE_REPO, CHOOSER_DIR, JUDGE_DIR, PRIVATE=1
set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN=hf_... (write token)}"

HF_USER=${HF_USER:-hilabs}
CHOOSER_REPO=${CHOOSER_REPO:-kev-4b-clinical-mapping}
JUDGE_REPO=${JUDGE_REPO:-ultrabert-clinical-judge}
CHOOSER_DIR=${CHOOSER_DIR:-$HOME/hilabs/kev-try/runs/kev_ub5_canon}
JUDGE_DIR=${JUDGE_DIR:-$HOME/hilabs/opencode-hilabs/artifacts/ultrabert_judge}
VIS=${VIS:-public}
[ "${PRIVATE:-0}" = "1" ] && VIS=private

hf auth login --token "$HF_TOKEN"

# 1) Kev chooser: LoRA adapter + pointer head (+ tokenizer). Loadable directly by Kev.
hf repo create "$HF_USER/$CHOOSER_REPO" --type model --"$VIS" --exist-ok
hf upload "$HF_USER/$CHOOSER_REPO" "$CHOOSER_DIR" . --repo-type model

# 2) Fine-tuned UltraBERT pairwise judge (~2 GB, resumable upload).
if [ -d "$JUDGE_DIR" ]; then
  hf repo create "$HF_USER/$JUDGE_REPO" --type model --"$VIS" --exist-ok
  hf upload-large-folder "$HF_USER/$JUDGE_REPO" "$JUDGE_DIR" --repo-type model
else
  echo "judge dir not found: $JUDGE_DIR (skipped)"
fi

echo "Published:"
echo "  https://huggingface.co/$HF_USER/$CHOOSER_REPO"
[ -d "$JUDGE_DIR" ] && echo "  https://huggingface.co/$HF_USER/$JUDGE_REPO"
