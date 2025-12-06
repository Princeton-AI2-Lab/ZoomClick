#!/bin/bash
#SBATCH --job-name=collect-sszoom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
set -x

USER=${USER:-sl8264}

# === paths ===
SCRATCH=/scratch/gpfs/$USER/zhiyuan
HOME_DIR=/home/$USER/zhiyuan
CODE_DIR=$HOME_DIR/code/zoomclick/UI-Venus
SRC_DIR=$HOME_DIR/code/zoomclick/UI-TARS
DATA_DIR=$SCRATCH/datasets/ScreenSpot-Zoom
LOG_DIR=$CODE_DIR/results/sszoom

# === cache on scratch ===
export HF_HOME=$SCRATCH/cache/hf
# export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=$SCRATCH/cache/torch
mkdir -p "$TRANSFORMERS_CACHE" "$TORCH_HOME"

# give a default value to OMP threads, avoid undefined error
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# === use python in environment (not activate, not depend on conda shell) ===
PY=/scratch/gpfs/$USER/zhiyuan/conda/envs/uivenus/bin/python
if [ ! -x "$PY" ]; then
  echo "[ERR] Python not found at $PY" >&2
  exit 127
fi

# all paths must exist
for p in "$CODE_DIR" "$DATA_DIR/images" "$DATA_DIR/annotations"; do
  [ -e "$p" ] || { echo "[ERR] Missing path: $p" >&2; exit 2; }
done

which "$PY" || true
"$PY" -V || true
nvidia-smi || true

cd "$CODE_DIR"
"$PY" grounding/collect_sszoom_accuracy.py \
    --results $SRC_DIR/results/sspro/qwen25_7b_in_depth_1.json \
    --dataset $DATA_DIR \
    --output $LOG_DIR/qwen25_7b_in_depth_1.json

"$PY" grounding/collect_sszoom_accuracy.py \
    --results $SRC_DIR/results/sspro/qwen25_7b_in_depth_2.json \
    --dataset $DATA_DIR \
    --output $LOG_DIR/qwen25_7b_in_depth_2.json

"$PY" grounding/collect_sszoom_accuracy.py \
    --results $SRC_DIR/results/sspro/qwen25_7b_in_depth_3.json \
    --dataset $DATA_DIR \
    --output $LOG_DIR/qwen25_7b_in_depth_3.json

"$PY" grounding/collect_sszoom_accuracy.py \
    --results $SRC_DIR/results/sspro/qwen25_7b_in_depth_4.json \
    --dataset $DATA_DIR \
    --output $LOG_DIR/qwen25_7b_in_depth_4.json

echo "Successfully Built ScreenSpot-Zoom!"
