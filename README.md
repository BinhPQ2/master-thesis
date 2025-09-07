# Master Thesis

This repository contains code and resources for my master thesis project.

## Environment

- **CUDA version:** 12.9

## PyTorch Installation

To install the correct version of PyTorch for CUDA 12.9, run:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

## Code Formatting

To keep the code clean and consistent, use [Black](https://black.readthedocs.io/en/stable/getting_started.html):

```bash
black {source_file_or_directory}
```
