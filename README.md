# UniIR for Pyserini

[**🌐 Homepage**](https://tiger-ai-lab.github.io/UniIR/) | [**🤗 Dataset(M-BEIR Benchmark)**](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) | [**🤗 Checkpoints(UniIR models)**](https://huggingface.co/TIGER-Lab/UniIR) | [**📖 arXiv**](https://arxiv.org/pdf/2311.17136.pdf) | [**Original UniIR GitHub**](https://github.com/TIGER-AI-Lab/UniIR)

This repository contains a fork of the original UniIR codebase, modified for easy [Pyserini](https://github.com/castorini/pyserini/) integration and repackaged as a PyPI package.

`current_version = "0.1.0"`

## 📦 Installation

Install the package directly from PyPI:

```bash
pip install uniir_for_pyserini
```

Or, install from source:

```bash
git clone https://github.com/castorini/UniIR-for-Pyserini.git
cd UniIR-for-Pyserini
pip install .
```

## Available Models

This package supports the following UniIR models from the [TIGER-Lab UniIR Hugging Face Hub](https://huggingface.co/TIGER-Lab/UniIR):

- `clip_sf_large`
- `blip_ff_large`

## Contact

For contact regarding the Pyserini integration section, please email [Sahel Sharifymoghaddam](sahel.sharifymoghaddam@uwaterloo.ca) or [Daniel Guo](daniel168.guo@gmail.com).

For contact regarding the original UniIR codebase, please email the authors of the original UniIR repository.

## Citation

If you use this work, please cite the original UniIR paper:

```bibtex
@article{wei2023uniir,
  title={Uniir: Training and benchmarking universal multimodal information retrievers},
  author={Wei, Cong and Chen, Yang and Chen, Haonan and Hu, Hexiang and Zhang, Ge and Fu, Jie and Ritter, Alan and Chen, Wenhu},
  journal={arXiv preprint arXiv:2311.17136},
  year={2023}
}
```

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
