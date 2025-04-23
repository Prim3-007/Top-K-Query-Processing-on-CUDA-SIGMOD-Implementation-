# Top-K Query Processing on GPU

This repository contains a Jupyter notebook implementing and benchmarking Top-K query processing algorithms on GPU using CuPy and CUDA. It includes both bitonic and sorting-based approaches with detailed performance evaluations.

---

## 📘 Notebook

- **`main.ipynb`** — The core notebook where Top-K logic is implemented and tested

---

## 🛠️ Features

- Bitonic Top-K using shared memory and CUDA kernels
- Sorting-based Top-K using `cp.sort()`
- Execution time benchmarks
- CuPy-powered GPU acceleration
- Synthetic dataset generation for testing
- (Optional) filtering and validation logic

---

## 🚀 How to Run

1. Open the notebook in Google Colab or any JupyterLab instance with CUDA support.
2. Make sure `cupy` and `matplotlib` are installed.
3. Run each cell sequentially and observe the outputs and plots.

---

## 📈 Output

- Average execution time per method (Sorting vs Bitonic)
- Speedup plots
- Top-K values from synthetic distributions

---

## 📘 Reference

This implementation is inspired by:
- SIGMOD 2018: *Efficient Top-K Query Processing on Massively Parallel Hardware*

---

## 📄 License

MIT License
