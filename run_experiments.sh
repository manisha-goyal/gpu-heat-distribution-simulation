#!/bin/bash

# Parameters for the experiment
DIMENSIONS=(100 500 1000 10000)      # Grid sizes
ITERATIONS=(50 100)                  # Number of iterations
BLOCK_SIZES=(16 32)                  # Block sizes to test (threads per block)
GRID_SIZES=(256 512 1024)            # Grid sizes to test (blocks per grid)
NUM_RUNS=5                           # Number of runs for averaging

# Output CSV file
output_file="experiment_results.csv"

# Write the header to the CSV file
echo "block_size,grid_size,iterations,grid_dimension,average_cpu_time,average_gpu_time,speedup" > $output_file

# Compile the program
nvcc -o heatdist heatdist.cu

echo "Starting experiment..."

# Loop over the specified block sizes, grid sizes, iterations, and dimensions
for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
    for GRID_SIZE in "${GRID_SIZES[@]}"; do
        for ITER in "${ITERATIONS[@]}"; do
            for DIM in "${DIMENSIONS[@]}"; do
                echo "Running for Block Size: $BLOCK_SIZE, Grid Size: $GRID_SIZE, Iterations: $ITER, Dimension: $DIM"

                # Initialize total times for averaging
                total_time_cpu=0
                total_time_gpu=0

                # Run the CPU and GPU versions multiple times for averaging
                for ((i=1; i<=$NUM_RUNS; i++)); do
                    # Run the heatdist program for CPU, capture the time taken
                    time_cpu=$(./heatdist $DIM $ITER 0 $BLOCK_SIZE $GRID_SIZE | grep "Time taken" | awk '{print $4}')
                    total_time_cpu=$(echo "$total_time_cpu + $time_cpu" | bc)

                    # Run the heatdist program for GPU, capture the time taken
                    time_gpu=$(./heatdist $DIM $ITER 1 $BLOCK_SIZE $GRID_SIZE | grep "Time taken" | awk '{print $4}')
                    total_time_gpu=$(echo "$total_time_gpu + $time_gpu" | bc)
                done

                # Calculate average times for CPU and GPU
                avg_time_cpu=$(echo "scale=6; $total_time_cpu / $NUM_RUNS" | bc)
                avg_time_gpu=$(echo "scale=6; $total_time_gpu / $NUM_RUNS" | bc)

                # Calculate the speedup (CPU time divided by GPU time)
                speedup=$(echo "scale=6; $avg_time_cpu / $avg_time_gpu" | bc)

                # Write the results to the CSV file using printf for formatting
                printf "%s,%s,%s,%s,%.6f,%.6f,%.6f\n" "$BLOCK_SIZE" "$GRID_SIZE" "$ITER" "$DIM" "$avg_time_cpu" "$avg_time_gpu" "$speedup" >> $output_file

                echo "Results - Block Size: $BLOCK_SIZE, Grid Size: $GRID_SIZE, Iterations: $ITER, Dimension: $DIM, Avg CPU Time: $avg_time_cpu, Avg GPU Time: $avg_time_gpu, Speedup: $speedup"
            done
        done
    done
done

echo "Experiment completed! Results saved in $output_file"