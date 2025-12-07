#!/bin/bash
#SBATCH --job-name=build-guizoom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
set -x

USER_NAME=${USER:-your_user_name}

# === paths (customize these for your environment) ===
SCRATCH=${SCRATCH:-/path/to/dataset}
HOME_DIR=${HOME_DIR:-/path/to/code}
CODE_DIR=${CODE_DIR:-${HOME_DIR}/ZoomClick}

DATA_DIR=${DATA_DIR:-${SCRATCH}/ScreenSpot-Pro}
OUT_DIR=${OUT_DIR:-${SCRATCH}/GUIZoom-Bench}
LOG_DIR=${LOG_DIR:-${CODE_DIR}/results/sspro}

# === cache on scratch (optional) ===
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

# build GUIZoom-Bench (ScreenSpot-Zoom reorganized)
"${PY}" GUIZoom-Bench/build_guizoom.py \
  --src_dataset_root "${DATA_DIR}" \
  --depth1 "${LOG_DIR}/venus_72b_depth_1.json" \
  --depth2 "${LOG_DIR}/venus_72b_in_depth_2.json" \
  --depth3 "${LOG_DIR}/venus_72b_in_depth_3.json" \
  --depth4 "${LOG_DIR}/venus_72b_in_depth_4.json" \
  --out_dir "${OUT_DIR}"

echo "Successfully built GUIZoom-Bench!"