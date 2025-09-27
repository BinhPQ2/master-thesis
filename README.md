# Master Thesis

This repository contains code and resources for my master thesis project.

## Environment

- **CUDA version:** 12.9

## PyTorch Installation

Basic Conda environment:

```
conda create -n paddleocr python=3.10 notebook
```

To install the correct version of PyTorch for CUDA 12.9, run:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

Install PaddleOCR:

```
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
pip install paddleocr==3.1.1
pip uninstall paddlex && pip install paddlex==3.1.1
pip install -r paddleocr_requirements.txt
```

## Code Formatting

To keep the code clean and consistent, use [Black](https://black.readthedocs.io/en/stable/getting_started.html):

```
black {source_file_or_directory}
```
