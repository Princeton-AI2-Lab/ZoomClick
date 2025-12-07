#!/usr/bin/env bash
#SBATCH --job-name=uivenus-zoomclick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --qos=high

set -euo pipefail
set -x

USER_NAME=${USER:-your_user_name}

# === Paths (customize for your environment) ===
# CODE_DIR: this ZoomClick repository (must contain grounding/eval_sspro_zoomclick.py)
CODE_DIR=${CODE_DIR:-/path/to/code/ZoomClick}

# DATA_DIR: ScreenSpot-Pro dataset root (must contain images/ and annotations/)
DATA_DIR=${DATA_DIR:-/path/to/dataset/ScreenSpot-Pro}

# MODEL_DIR: UI-Venus-Ground-7B model weights
MODEL_DIR=${MODEL_DIR:-/path/to/models/UI-Venus-Ground-7B}

# LOG_DIR: directory to save evaluation results
LOG_DIR=${LOG_DIR:-${CODE_DIR}/results/sspro}
mkdir -p "${LOG_DIR}"

# === Optional caches ===
SCRATCH=${SCRATCH:-/path/to/scratch}
export HF_HOME=${HF_HOME:-${SCRATCH}/cache/hf}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_HOME=${TORCH_HOME:-${SCRATCH}/cache/torch}
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# === Python in conda env (zoomclick) ===
PY=${PY:-/path/to/conda/envs/zoomclick/bin/python}
[ -x "${PY}" ] || { echo "[ERR] Python not found at ${PY}" >&2; exit 127; }

# Sanity checks
for p in "${CODE_DIR}" "${DATA_DIR}/images" "${DATA_DIR}/annotations" "${MODEL_DIR}"; do
  [ -e "$p" ] || { echo "[ERR] Missing path: $p" >&2; exit 2; }
done

cd "${CODE_DIR}"

# Run evaluation: ZoomClick + UI-Venus-Ground-7B on ScreenSpot-Pro
"${PY}" grounding/eval_sspro_zoomclick.py \
  --backend uivenus \
  --model_type ui_venus_ground_7b \
  --model_name_or_path "${MODEL_DIR}" \
  --screenspot_imgs "${DATA_DIR}/images" \
  --screenspot_test "${DATA_DIR}/annotations" \
  --task "all" \
  --inst_style "instruction" \
  --language "en" \
  --gt_type "positive" \
  --log_path "${LOG_DIR}/zoomclick_venus_7b_clip.json" \
  --in_depth 3 \
  --in_ratio 0.5 \
  --in_min_crop 768 \
  --patch_size 2 \
  --center_mode "clip" \
  --prezoom_px_thresh 50

echo "Evaluation finished."