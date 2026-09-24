# Environment setup

TRACE_hic uses Python 3.10 or newer. Its Python dependencies and supported version ranges are declared in [`pyproject.toml`](../pyproject.toml). The commands below use a virtual environment and `pip` on Linux; run them from the repository root.

## 1. Create an environment

```bash
cd /path/to/TRACE_hic
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If `venv` is unavailable on Ubuntu, install the matching `python3-venv` system package and retry. Activate `.venv` again in each new shell before running TRACE_hic commands.

## 2. Install PyTorch and TRACE_hic

Training requires a CUDA-capable NVIDIA GPU. Before installing the project, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose **Linux → Pip → Python → a CUDA build compatible with your NVIDIA driver**, then run the command it provides inside `.venv`. The project requires `torch>=2.5,<3`. Installing PyTorch first lets you choose the correct build for your machine.

For CPU-only setup, choose **CPU** in the same selector. CPU is suitable for setup checks and may be used for prediction with `--device cpu --precision float32`; the training entry point requires a GPU.

Then install the project and its remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` installs the local source in editable mode. Dependency names and version ranges are maintained in `pyproject.toml`, so the two files cannot drift apart. The package includes PyTorch Lightning, NumPy, pandas, SciPy, scikit-image, h5py, pyBigWig, and Cooler. You do not need to install these separately.

## 3. Check the installation

```bash
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available())"
trace-hic-train --help
trace-hic-predict --help
```

For GPU training, `torch.cuda.is_available()` must print `True`. If it prints `False`, check the NVIDIA driver with `nvidia-smi`, then confirm that the installed PyTorch build matches the driver using the [PyTorch installation guide](https://pytorch.org/get-started/locally/). A CPU build of PyTorch will not use the GPU.

To run the repository's small tests, install the optional test dependency and run:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

These checks validate the installation and input contracts; they do not reproduce a full training run.

## 4. Supply model weights and data

Installation does not download the hg38 FASTA, RNA-seq data, Hi-C labels, or model weights. See the [README](../README.md#prepare-input-data) for the expected data layout and preparation commands.

Put trusted model files at `weights/stage1.pt` (SUCCEED) and `weights/stage2.ckpt` (TRACE_hic), or pass explicit checkpoint paths to the commands. Check their SHA256 hashes against [`artifacts.json`](../artifacts.json). A new Hi-C head training run needs only `stage1.pt`; prediction needs both weights. These files are ignored by Git.

The training command defaults to `--devices 2 --precision bf16-mixed` and uses the GPU accelerator. Set `--devices` to the number of GPUs you actually have; BF16 requires compatible hardware. Prediction defaults to `--device cuda:0 --precision bf16`. On a CPU-only system, add `--device cpu --precision float32` to the prediction command. Full-window prediction can require substantial RAM or GPU memory.
