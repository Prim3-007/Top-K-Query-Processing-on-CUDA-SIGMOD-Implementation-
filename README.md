# Top-K GPU Query Processing

This project implements GPU-accelerated Top-K query processing using CuPy and CUDA, showcasing advanced performance optimization techniques from the SIGMOD 2018 paper *"Efficient Top-K Query Processing on Massively Parallel Hardware."*

---

## 📘 Notebook Overview

**File:** `main.ipynb`

This notebook includes:
- Implementation of Bitonic Top-K using fused CUDA kernels
- Baseline sorting-based Top-K via CuPy
- Optional fused filtering (e.g., `WHERE value > threshold`)
- Accuracy checking between methods
- GPU memory usage tracking (`memGetInfo`)
- Runtime benchmarking across multiple `K` values and distributions
- Speedup visualization and analysis

---

## 🛠️ Features

- ✅ GPU-based Bitonic Top-K kernel
- ✅ Sorting-Based Top-K using `cp.sort()`
- ✅ Optional data filtering before Top-K
- ✅ Accuracy validation using `np.allclose`
- ✅ Multi-distribution data testing
- ✅ Memory profiling before/after GPU calls
- ✅ Streamlined for Google Colab or local CUDA environments

---

## 🚀 How to Run

1. Open `main.ipynb` in [Google Colab](https://colab.research.google.com) or a local Jupyter environment with GPU support.
2. Make sure `cupy`, `matplotlib`, `seaborn`, and `numpy` are available.
3. Run the notebook sequentially and review speedup plots and profiling outputs.

---

## 📈 Sample Outputs

- Runtime benchmarks for each method
- GPU memory usage statistics
- Accuracy comparison table
- Distribution-aware performance graphs

---

## 🧠 Reference

- **T. Kipf, A. Kemper, T. Neumann.**  
  *Efficient Top-K Query Processing on Massively Parallel Hardware* (SIGMOD 2018)  
  [Link to Paper](https://dl.acm.org/doi/10.1145/3183713.3196909)

---

## 📄 License

MIT License. Use for research, experimentation, or educational purposes.
