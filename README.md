<h1 align="center">🔍 Fine-Grained Pill Classification</h1>
<h3 align="center">Enhanced Few-Shot Pill Identification with Coordinate Attention and Domain Adaptation</h3>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.6-blue.svg"/></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-1.9.0-orange.svg"/></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green.svg"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Platform-Azure%20ML-blue.svg"/></a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Training](#training)
- [Inference & Demo](#inference--demo)
- [Results](#results)
- [Citation](#citation)

---

## Overview

A fine-grained pill identification system built on the [ePillID benchmark](https://github.com/usuyama/ePillID-benchmark), enhanced with **Coordinate Attention**, **Domain Adaptation via Gradient Reversal Layer (GRL)**, and a production-ready **FastAPI** inference server.

This project addresses the challenge of identifying pharmaceutical pills from consumer-captured images. The core difficulty lies in the large **domain gap** between high-quality reference images and real-world consumer photos (varying lighting, angles, backgrounds).

The system uses a **multi-task learning** approach combining:

- Metric learning (contrastive + triplet loss) to learn discriminative embeddings
- Classification heads (Cross-Entropy + ArcFace) for direct class prediction
- Domain Adaptation to bridge the reference–consumer domain gap

---

## Key Features

| Feature | Description |
|---|---|
| 🔍 **Coordinate Attention** | Spatial attention module (CVPR 2021) focusing on pill imprints and shape |
| 🔄 **Gradient Reversal Layer** | Domain-invariant feature learning between reference and consumer images |
| ⚡ **Fast MPN-COV Pooling** | Support for GAvP, MPNCOV, BCNN, CBP pooling strategies |
| 🏷️ **ArcFace Head** | Additive angular margin loss for better class separability |
| 🌐 **FastAPI Server** | Production-ready REST API with NIH RxNav + OpenFDA integration |
| ☁️ **Azure ML** | Full training pipeline with cross-validation and run tracking |

---

## Architecture
```
Input Image (224×224)
        │
        ▼
  ResNet50 Backbone
        │
        ▼
 Coordinate Attention
        │
        ▼
   GAvP Pooling → [2048-dim vector]
        │
   ┌────┴────┐
   │         │
   ▼         ▼
Embedding  Domain Classifier
 Layer      (w/ GRL)
   │
   ├──► BinaryHead (CE Loss)
   ├──► MarginHead (ArcFace Loss)
   ├──► Contrastive Loss
   ├──► Triplet Loss
   └──► Domain Loss
```

---

## Project Structure
```
├── src/
│   ├── models/
│   │   ├── enhanced_embedding_model.py
│   │   ├── enhanced_multihead_model.py
        ├── enhanced_multihead_trainer.py
│   │   ├── coordinate_attention.py
│   │   ├── grl_domain_classifier.py
│   │   ├── enhanced_losses.py
│   │   └── fast-MPN-COV/
│   ├── train_cv.py
│   ├── train_nocv.py
    ├── arguments.py
│   ├── multihead_trainer.py
│   └── metric_test_eval.py
│
├── ePillID_demo/
│   ├── main.py
│   ├── mapping_utils.py
│   ├── classes.txt
│   └── drug_dict.json
│
├── docker/
├── Setup_env.txt
└── README.md
```

---

## Environment Setup (Setup.txt)

### 1. Install Miniconda & Create Environment
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

conda create -n epillid python=3.6 -y
conda activate epillid
```

### 2. Install PyTorch (CUDA 11.1)
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    --extra-index-url https://download.pytorch.org/whl/cu111
```

### 3. Install Dependencies
```bash
pip install pandas tqdm azureml azureml.core
pip install scikit-learn==0.22.2
conda install -c conda-forge opencv
conda install -c conda-forge imgaug
```

### 4. System Libraries
```bash
sudo apt-get update && sudo apt-get install -y libtiff5-dev
sudo apt update && sudo apt install -y libsm6 libxext6 libxrender-dev
conda install -c conda-forge libtiff=4.4.0
```

---

## Training
```bash
export AZUREML_DEPRECATE_WARNING=False
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

nohup python src/train_cv.py \
    --appearance_network resnet50 \
    --pooling GAvP \
    --max_epochs 100 \
    --data_root_dir ePillID_data \
    --batch_size 64 \
    --results_dir ./outputs/Resnet50_GAVP_CV \
    --num_workers 0 \
    --add_persp_aug 1 \
    2>&1 > ./run_$(date +%Y%m%d_%H%M).log &
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--appearance_network` | `resnet50` | Backbone architecture |
| `--pooling` | `GAvP` | Pooling method |
| `--batch_size` | `64` | Batch size |
| `--max_epochs` | `100` | Maximum training epochs |
| `--init_lr` | `1e-4` | Initial learning rate |
| `--add_persp_aug` | `1` | Enable perspective augmentation |

---

## Inference & Demo

### Start the FastAPI Server
```bash
cd ePillID_demo
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open API docs at: `http://127.0.0.1:8000/docs`

### API Endpoint
```
POST /predict
Content-Type: multipart/form-data
Body: file=<pill_image>
```

**Example Response:**
```json
{
  "status": "success",
  "pill_info": {
    "pill_id": "00093-7156",
    "brand_name": "Simvastatin"
  },
  "display_quick": {
    "usage": "Used to lower cholesterol...",
    "dosage": "Take once daily in the evening...",
    "warnings": "Report any unexplained muscle pain..."
  },
  "confidence": "87.34%"
}
```

---

## Results

| Model | Pooling | mAP | GAP | Top-1 Acc |
|---|---|---|---|---|
| ResNet50 baseline | GAvP | 94.02 ± 0.66 | 78.27 ± 2.35  | - |
| ResNet50 + CA + GRL | BCNN | 94.92 ± 0.41 | 78.93 ± 1.31  | 90.54 ± 0.93 |


---

## Citation
```bibtex
@inproceedings{usuyama2020epillid,
  title     = {ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification},
  author    = {Usuyama, Naoto and Bagdanov, Andrew D.},
  booktitle = {CVPR Workshops},
  year      = {2020}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
