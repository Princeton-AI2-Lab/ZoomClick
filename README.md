# Zoom in, Click out: Unlocking and Evaluating the Potential of Zooming for GUI Grounding

<p>
  <a href="https://github.com/zhiyuanjiang04">Zhiyuan Jiang</a><sup>*</sup> ·
  <a href="https://github.com/shxie2020">Shenghao Xie</a><sup>*</sup> ·
   Wenyi Li ·
   Wenqiang Zu ·
   Peihang Li · 
   Jiahao Qiu ·
   Siqi Pei ·
   Lei Ma<sup>†</sup> ·
   Tiejun Huang ·
   Mengdi Wang<sup>†</sup> ·
  <a href="https://github.com/SlongLiu">Shilong Liu</a><sup>†</sup>
</p>

<p>
  <em>* Equal contribution &nbsp;·&nbsp; † Corresponding authors</em>
</p>

This repo provides the official implementaion for ZoomClick and GUIZoom-Bench. We first release the implementations of our method based on UI-Venus and Qwen3-VL, together with the dataset re-organization method of our benchmark.

<img width="1924" height="836" alt="fig1" src="https://github.com/user-attachments/assets/2243c3a7-8465-4117-9312-274c41b3f46d" />


## Highlights
- **ZoomClick**：Our training-free method treats zoom as a strong prior and explicitly models zoom–locate–click interaction on high-resolution GUIs, decomposing GUI grounding into a sequence of reliable local decisions and substantially improving robustness and accuracy for both general vision–language and specialized GUI grounding models.
- **GUIZoom-Bench**：Our benchmark is dedicated to evaluating models’ adaptability to zoom in GUI grounding, focusing on scenarios that require dynamic spatial focusing, adaptive context switching, and fine-grained element localization, thus providing a standardized testbed for zoom-based training and test-time scaling.
- **Strong Performance**：With ZoomClick, UI-Venus-72B achieves a 73.1% success rate on ScreenSpot-Pro, establishing new state-of-the-art performance on this mainstream GUI grounding benchmark.


## Repository Structure

- **`grounding/`**: Evaluation scripts for ZoomClick
  - `eval_sspro_zoomclick.py`: Main script to evaluate ZoomClick on ScreenSpot-Pro.
  - `models/`: Backbone wrappers and ZoomClick variants (Qwen3-VL, UI-Venus).

- **`GUIZoom-Bench/`**: Scripts for building and evaluating GUIZoom-Bench
  - `build_guizoom.py`: Re-organize ScreenSpot-Pro–style data into GUIZoom-Bench.
  - `collect_guizoom_accuracy.py`: Compute accuracy and related metrics on GUIZoom-Bench.

- **`results/`**: Example JSON results used to reproduce tables and figures
  - `sspro/venus_72b.json`, `sspro/venus_72b_depth_1.json`, etc.

- **`scripts/`**: Utility and cluster (Slurm) scripts
  - `run_zoomclick_qwen3.slurm`, `run_zoomclick_uivenus.slurm`: Example Slurm jobs for running ZoomClick evaluations and benchmark building.

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
       This will create GUIZoom-Bench splits, annotations, images, and statistics under `/path/to/dataset/GUIZoom-Bench`.

## Evaluation
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

2. **Eval on GUIZoom-Bench**:
   - Directly follow the same commands as in Eval on ScreenSpot-Pro, but set `DATA_DIR=${SCRATCH}/datasets/GUIZoom-Bench` instead of `DATA_DIR=${SCRATCH}/datasets/ScreenSpot-Pro`.

## Citation
If you find our work helpful, please leave us a star and cite our paper. Thank you!
