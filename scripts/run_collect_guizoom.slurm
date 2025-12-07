#!/bin/bash
#SBATCH --job-name=collect-guizoom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
set -x

USER_NAME=${USER:-your_user_name}

# === paths (customize for your environment) ===
# CODE_DIR: this ZoomClick repo
CODE_DIR=${CODE_DIR:-/path/to/code/ZoomClick}

# SRC_DIR: directory that contains the original ScreenSpot-Pro evaluation results
SRC_DIR=${SRC_DIR:-/path/to/src_results}

# DATA_DIR: GUIZoom-Bench dataset (reorganized from ScreenSpot-Pro)
DATA_DIR=${DATA_DIR:-/path/to/dataset/GUIZoom-Bench}

# LOG_DIR: where to save collected GUIZoom-Bench metrics
LOG_DIR=${LOG_DIR:-${CODE_DIR}/results/guizoom}
mkdir -p "${LOG_DIR}"

# === optional caches ===
SCRATCH=${SCRATCH:-/path/to/scratch}
export HF_HOME=${HF_HOME:-${SCRATCH}/cache/hf}
export TORCH_HOME=${TORCH_HOME:-${SCRATCH}/cache/torch}
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

# threads & env flags
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# === python in conda env (zoomclick) ===
PY=${PY:-/path/to/conda/envs/zoomclick/bin/python}
if [ ! -x "${PY}" ]; then
  echo "[ERR] Python not found at ${PY}" >&2
  exit 127
fi

# sanity checks
for p in "${CODE_DIR}" "${DATA_DIR}/images" "${DATA_DIR}/annotations"; do
  [ -e "$p" ] || { echo "[ERR] Missing path: $p" >&2; exit 2; }
done

which "${PY}" || true
"${PY}" -V || true

cd "${CODE_DIR}"

# Collect accuracy on GUIZoom-Bench for different depth results
"${PY}" GUIZoom-Bench/collect_guizoom_accuracy.py \
  --results "${SRC_DIR}/results/sspro/venus_72b_in_depth_1.json" \
  --dataset "${DATA_DIR}" \
  --output  "${LOG_DIR}/venus_72b_in_depth_1.json"

"${PY}" GUIZoom-Bench/collect_guizoom_accuracy.py \
  --results "${SRC_DIR}/results/sspro/venus_72b_in_depth_2.json" \
  --dataset "${DATA_DIR}" \
  --output  "${LOG_DIR}/venus_72b_in_depth_2.json"

"${PY}" GUIZoom-Bench/collect_guizoom_accuracy.py \
  --results "${SRC_DIR}/results/sspro/venus_72b_in_depth_3.json" \
  --dataset "${DATA_DIR}" \
  --output  "${LOG_DIR}/venus_72b_in_depth_3.json"

"${PY}" GUIZoom-Bench/collect_guizoom_accuracy.py \
  --results "${SRC_DIR}/results/sspro/venus_72b_in_depth_4.json" \
  --dataset "${DATA_DIR}" \
  --output  "${LOG_DIR}/venus_72b_in_depth_4.json"

echo "Successfully collected GUIZoom-Bench metrics!"