# Efficient Top-K Query Processing on GPU

This repository implements and benchmarks GPU-accelerated algorithms for extracting Top-K elements from large datasets, based on the SIGMOD 2018 paper:  
**"Efficient Top-K Query Processing on Massively Parallel Hardware"**  
Includes Bitonic Top-K, Sorting-Based Top-K, memory profiling, accuracy checks, and fused filtering.

---

## 📂 Contents

- `Final_Optimized_TopK_Colab.ipynb` — Main Colab notebook with modular code
- Bitonic Top-K implementation using CUDA via CuPy
- Sorting-based Top-K using CuPy's optimized radix sort
- Filtered Top-K (e.g., retweet_count > threshold)
- Accuracy validation: compare Top-K results between methods
- Memory profiling via `cupy.cuda.runtime.memGetInfo()`
- Multi-distribution data testing (uniform, exponential, bucket killer, etc.)

---

## 🚀 Features

- ✅ Full Bitonic Top-K with kernel fusion
- ✅ Sorting-based Top-K as baseline
- ✅ Optional fused filter stage inside Bitonic pipeline
- ✅ Memory profiling before/after each GPU kernel
- ✅ Accuracy verification using `np.allclose`
- ✅ Support for multiple synthetic data distributions
- ✅ Execution time + speedup plots

---

## 📊 Usage

Run the notebook in Google Colab or any Jupyter environment with CUDA support.

```bash
# Example: Generate and compare Top-K performance
K = [64, 128, 256, 512]
distribution = "exponential"
threshold = 50  # Filter for fused filtering
```

---

## 📈 Sample Output

- Speedup graphs
- Accuracy table for each K
- Memory used before/after kernels

---

## 📘 Reference

- T. Kipf, A. Kemper, T. Neumann.  
  **"Efficient Top-K Query Processing on Massively Parallel Hardware."**  
  *SIGMOD 2018* [[PDF](https://dl.acm.org/doi/10.1145/3183713.3196909)]

---

## 📄 License

MIT License. Feel free to use and cite.
