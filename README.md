# Zoom in, Click out: Unlocking and Evaluating the Potential of Zooming for GUI Grounding

[![arXiv](https://img.shields.io/badge/arXiv-2512.05941-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.05941)

<p>
  <a href="https://github.com/zhiyuanjiang04">Zhiyuan Jiang</a><sup>*</sup> ·
  <a href="https://github.com/shxie2020">Shenghao Xie</a><sup>*</sup> ·
  <a href="https://github.com/wenyi-li">Wenyi Li</a> ·
  <a href="https://github.com/zuwenqiang">Wenqiang Zu</a> ·
  <a href="https://github.com/abrohamLee">Peihang Li</a> · 
  <a href="https://github.com/CharlesQ9">Jiahao Qiu</a> ·
  <a href="https://siqi-pei.github.io">Siqi Pei</a> ·
  <a href="https://www.ai.pku.edu.cn/en/info/1462/2126.htm">Lei Ma</a><sup>†</sup> ·
  <a href="https://cfcs.pku.edu.cn/english/people/faculty/tiejunhuang/index.htm">Tiejun Huang</a> ·
  <a href="https://mwang-delta.vercel.app/">Mengdi Wang</a><sup>†</sup> ·
  <a href="https://github.com/SlongLiu">Shilong Liu</a><sup>†</sup>
</p>

<p>
  <em>* Equal contribution &nbsp;·&nbsp; † Corresponding authors</em>
</p>

This repo provides the official implementaion for ZoomClick and GUIZoom-Bench.

<img width="1924" height="836" alt="fig1" src="https://github.com/user-attachments/assets/2243c3a7-8465-4117-9312-274c41b3f46d" />


## Highlights
- **ZoomClick**：Exploring zooming to dig out more grounding priors of both generalist VLMs and specialized GUI grounding models in a training-free, principled, effective way.
- **GUIZoom-Bench**：Evaluating models’ zoom capability with explainable standards, supporting future research on zoom-based training and test-time scaling.
- **Strong Performance**：With ZoomClick, UI-Venus-72B achieves a 73.1% success rate on ScreenSpot-Pro, establishing a new state-of-the-art performance.


## Repository Structure

- **`grounding/`**: Evaluation scripts for ZoomClick
  - `eval_sspro_zoomclick.py`: Main script to evaluate ZoomClick on ScreenSpot-Pro.
  - `models/`: Backbone wrappers and ZoomClick variants (Qwen3-VL, UI-Venus).

- **`GUIZoom-Bench/`**: Scripts for building and evaluating GUIZoom-Bench
  - `build_guizoom.py`: Re-organize ScreenSpot-Pro dataset into GUIZoom-Bench.
  - `collect_guizoom_accuracy.py`: Compute accuracy and related metrics on GUIZoom-Bench based on grounding results on Screenspot-Pro.

- **`results/sspro`**: Example JSON results used to reproduce tables and figures
  - `zoomclick_*_clip.json` files are the results provided by default settings in `run_zoomclick_*.slurm`.
  - `venus_7b_depth_(1-4).json` are results used to build GUIZoom-Bench.

- **`scripts/`**: Utility and cluster (Slurm) scripts
  - `run_zoomclick_*.slurm`: Example Slurm jobs for running ZoomClick evaluations on Screenspot-Pro.
  - `run_collect_guizoom.slurm`: Slurm script used for GUIZoom-Bench result re-organization.
  - `run_build_guizoom.slurm`: Slurm script used for building GUIZoom-Bench dataset.

## Installation
1. **Environment Setup**

   ```
   # Clone the repository
   git clone https://github.com/Princeton-AI2-Lab/ZoomClick.git
   cd ZoomClick

   # (Recommended) Create a conda environment
   conda create -n zoomclick python=3.10 -y
   conda activate zoomclick

   # Install dependencies: We are actively working on releasing a general, easy-to-use requirements file for this project.
   pip install -r requirements.txt
   ```
2. **Data Preparation**

   **Screenspot-Pro**
   - Download Screenspot-Pro from its official repository or dataset release: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding.
   - Recommended directory layout:
     ```
     /path/to/dataset/Screenspot-Pro/
       images/
       annotations/
     ```
     Set this path via `--data-root` (or equivalent) in `grounding/eval_sspro_*.py` or through command-line arguments.

    **GUIZoom-Bench**
    - GUIZoom-Bench is built from reorganization of Screenspot-Pro dataset.
        ```
        python GUIZoom-Bench/build_guizoom.py \
           --src_dataset_root /path/to/dataset/Screenspot-Pro \
           --depth1 /path/to/depth1.json \
           --depth2 /path/to/depth2.json \
           --depth3 /path/to/depth3.json \
           --depth4 /path/to/depth4.json \
           --out_dir /path/to/dataset/GUIZoom-Bench
        ```
        - args `--depthx`: 'results/sspro/venus_7b_depth_x'
    
       This will create GUIZoom-Bench splits, annotations, images, and statistics under `/path/to/dataset/GUIZoom-Bench`. We recommend to store Screenspot-Pro and GUIZoom-Bench in the same `dataset` directory for convenient use.

## Evaluation
We recommend using at least one A100 GPU for models up to 8B, and at least four A100 GPUs for models 32B and above.

1. **Eval on Screenspot-Pro**：
   - On a cluster: Modify Basic paths in `scripts/run_zoomclick_uivenus.slurm` and `scripts/run_zoomclick_qwen3.slurm` according to your data structure and submit the slurm script.
   - Otherwise:
     - Activate conda environment: `conda activate path/to/your/conda/envs/zoomclick`
     - Run evaluation according to your own setting:
         ```
         python grounding/eval_sspro_zoomclick.py \
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
         ```
         - `--in_depth`: The number of iterative zoom-in steps applied during evaluation.
         - `--in_ratio`: The shrink ratio for each zoom-in step.
         - `--in_min_crop`: The minimum crop size to retain sufficient visual context during zooming.
         - `--patch_size`: The grid resolution used when estimating the zoom center in Pre-Zoom. By default, `patch_size=2` represents a 2*2 grid.
         - `--center_mode`: Mode of boundary handling. Choose from `shift`, 'clip', 'shrink'.
         - `--prezoom_px_thresh`: Threshold of pixel distance used in Pre-zoom. Refer to our paper for more details.

2. **Eval on GUIZoom-Bench**:
   There are two ways to evaluate your model on GUIZoom-Bench:
     - **Re-organize Screenspot-Pro data**:
         - Build GUIZoom-Bench following commands in Data Preparation.
         - Directly follow the same commands as in Eval on ScreenSpot-Pro, but set `DATA_DIR=${SCRATCH}/datasets/GUIZoom-Bench` instead of `DATA_DIR=${SCRATCH}/datasets/ScreenSpot-Pro`.
     - **Re-organize Screenspot-Pro results** (Recommended):
         - Because our benchmark is built from Screenspot-Pro, evaluation results on Screenspot-Pro can also be re-organized into GUIZoom-Bench results without additional computation. This design removes the need to store a duplicated dataset for benchmarking, effectively reducing storage usage and avoiding waste.
         - Simply adjust arguments in `scripts/run_collect_guizoom.slurm`:
             - `--results`: path to the JSON result to be re-orgainzed
             - `--dataset`: path to GUIZoom-Bench
             - `--output`: path to the re-orgainzed JSON result

## Citation
If you find our work helpful, please leave us a star and cite our paper：

```
@misc{jiang2025zoominclickout,
      title={Zoom in, Click out: Unlocking and Evaluating the Potential of Zooming for GUI Grounding}, 
      author={Zhiyuan Jiang and Shenghao Xie and Wenyi Li and Wenqiang Zu and Peihang Li and Jiahao Qiu and Siqi Pei and Lei Ma and Tiejun Huang and Mengdi Wang and Shilong Liu},
      year={2025},
      eprint={2512.05941},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.05941}, 
}
```
