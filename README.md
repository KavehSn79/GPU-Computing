# GPU Computing Exercises (CUDA)

This repository contains a collection of CUDA programming exercises completed as part of a **GPU Computing course**.
The exercises progress from basic CUDA kernel development to advanced GPU optimization techniques, including parallel algorithms, memory optimization, streams, and Tensor Core programming.

The goal of these projects is to gain hands-on experience with **GPU architecture**, **parallel programming models**, and **performance optimization** on NVIDIA GPUs.

---

## 🧠 Topics Covered

* CUDA kernel programming
* Thread and block hierarchy
* GPU memory model (global, shared, registers)
* Performance benchmarking and profiling
* Parallel scan algorithms
* Asynchronous execution and CUDA streams
* Shared-memory tiling
* Tensor Cores (WMMA)
* Stencil-based physical simulations

---

## 📁 Exercises Overview

### **Exercise 1 — CUDA Matrix & Vector Computations**

**Description:**
Converted a sequential C++ matrix/vector computation into a parallel CUDA implementation and evaluated performance across different kernel launch configurations.

**Key Learnings:**

* Mapping computations to CUDA threads
* Kernel launch configuration tuning
* GPU benchmarking and performance analysis
* Understanding GPU occupancy and parallel execution

---

### **Exercise 2 — Parallel Inclusive Scan (Kogge–Stone)**

**Description:**
Implemented a parallel inclusive prefix scan for complex numbers using the Kogge–Stone algorithm, with complex multiplication as the scan operation.

**Key Learnings:**

* Parallel scan algorithms
* Warp divergence reduction
* Shared memory optimization
* Thread coarsening and memory coalescing

---

### **Exercise 3 — Streams & Tiled Matrix Multiplication**

**Description:**
Part 1 implements overlapping of communication and computation using CUDA streams.
Part 2 optimizes naive matrix multiplication using shared-memory tiling and profiling with NVIDIA Nsight.

**Key Learnings:**

* CUDA streams and asynchronous execution
* Overlapping memory transfers with computation
* Shared memory tiling strategies
* Performance-driven optimization using profiling tools

---

### **Exercise 4 — 1D Convolution with Tensor Cores (WMMA)**

**Description:**
Implemented a 1D convolution by reformulating it as a matrix multiplication and executing it using Tensor Core WMMA instructions.

**Key Learnings:**

* Convolution-to-matrix multiplication transformation
* WMMA API and Tensor Core usage
* Warp-level programming
* Hardware constraints of Tensor Cores

---

### **Exercise 5 — 2D Heat Flow Simulation**

**Description:**
Reimplemented a sequential 2D heat flow simulation as a CUDA-accelerated stencil computation using shared memory and double buffering.

**Key Learnings:**

* Stencil-based GPU computations
* Double buffering on the GPU
* Shared memory optimization
* Efficient handling of iterative simulations

---

## 🛠️ Tools & Technologies

* **CUDA / C++**
* **NVIDIA GPUs**
* **CUDA Streams**
* **Shared Memory & WMMA**
* **NVIDIA Nsight (profiling)**


