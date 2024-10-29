# Heat Distribution Simulation on GPU

This project explores the performance benefits of using GPUs for a heat distribution simulation, comparing execution times and speedups across various grid dimensions, thread, and block configurations. The experiments were run using CUDA to take advantage of parallel processing on an NVIDIA GeForce RTX 2080 Ti GPU.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
  - [Experiment 1: 50 Iterations](#experiment-1-50-iterations)
  - [Experiment 2: 100 Iterations](#experiment-2-100-iterations)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction
This project simulates heat distribution on a grid and demonstrates the speedups achieved when using GPU parallel processing over a CPU. The code was modified to accept grid dimensions, thread, and block configurations as inputs to experiment with performance across different setups.

## Project Structure
- **`heatdist.cu`**: Main CUDA file for running the heat distribution simulation on the GPU.
- **`run_experiments.sh`**: Shell script to automate experiments across various grid sizes, block, and thread configurations.
- **`experiment_results.csv`**: CSV file containing average timings for each configuration.
- **`generate_charts.py`**: Python script to generate visualizations from experiment results.

## Requirements
- **CUDA Toolkit**: Ensure CUDA is installed for compiling and running GPU code.
- **NVIDIA GPU**: This project is optimized for an NVIDIA GeForce RTX 2080 Ti.
- **Python**: For data analysis and plotting.
- **Python Packages**:
  - `pandas`
  - `matplotlib`

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/heat-distribution-gpu.git
    cd heat-distribution-gpu
    ```
2. Compile the CUDA code:
    ```bash
    nvcc -o heatdist heatdist.cu
    ```
3. Install Python dependencies:
    ```bash
    pip install pandas matplotlib
    ```

## Usage
To run the experiments and capture performance metrics:
1. Run the experiment script:
    ```bash
    bash run_experiments.sh
    ```
   This script runs the simulation for different grid sizes, block sizes, and thread configurations, and outputs results to `experiment_results.csv`.
   
2. Generate visualizations:
   After experiments are complete, use `generate_charts.py` to generate performance charts for each configuration.

## Experiments
### Experiment 1: 50 Iterations
For 50 iterations, this experiment compares GPU and CPU times across different configurations and grid dimensions. Results highlight the GPU’s parallelism efficiency for varying grid sizes.

### Experiment 2: 100 Iterations
This experiment doubles the workload to 100 iterations, offering additional insights, especially for larger grid dimensions.

## Results
- **Speedup**: GPU processing achieved significant speedup for larger grid sizes.
- **Optimal Configuration**: Based on our findings, a block size of 16 and a grid size of 512 gave the best overall performance across both 50 and 100 iterations.

Visual results are available in the `/results` directory, where speedup charts demonstrate GPU’s advantage at larger grid sizes.