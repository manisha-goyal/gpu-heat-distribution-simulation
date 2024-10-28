import pandas as pd
import matplotlib.pyplot as plt

# Read the experiment results from the CSV file
data = pd.read_csv('experiment_results.csv')

# Define a function to plot CPU vs GPU times and Speedup for each configuration
def plot_experiment(data, block_size, iterations):
    # Filter data for the given block size and iteration count
    filtered_data = data[(data['block_size'] == block_size) & (data['iterations'] == iterations)]
    dimensions = filtered_data['grid_dimension'].astype(str)
    
    # CPU vs GPU times plot
    plt.figure(figsize=(12, 6))
    cpu_times = filtered_data['average_cpu_time']
    gpu_times = filtered_data['average_gpu_time']
    
    # Define bar width and index positions
    bar_width = 0.35
    index = range(len(dimensions))
    
    # Plot the bars for CPU and GPU times
    cpu_bars = plt.bar(index, cpu_times, bar_width, label='CPU Time')
    gpu_bars = plt.bar([i + bar_width for i in index], gpu_times, bar_width, label='GPU Time')
    
    # Add labels above each bar for CPU and GPU times
    for bar in cpu_bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.6f}', va='bottom', ha='center', fontsize=8)
    for bar in gpu_bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.6f}', va='bottom', ha='center', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Grid Dimension (N)')
    plt.ylabel('Time (seconds)')
    plt.title(f'Performance for Block Size {block_size}, {iterations} Iterations')
    plt.xticks([i + bar_width / 2 for i in index], dimensions, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plot_block_{block_size}_iter_{iterations}_times.png')
    plt.show()
    
    # Speedup plot
    plt.figure(figsize=(12, 6))
    speedups = filtered_data['speedup']
    
    # Plot the bars for Speedup
    speedup_bars = plt.bar(index, speedups, bar_width, label='Speedup')
    
    # Add labels above each bar for Speedup values
    for bar in speedup_bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Grid Dimension (N)')
    plt.ylabel('Speedup (CPU Time / GPU Time)')
    plt.title(f'Speedup for Block Size {block_size}, {iterations} Iterations')
    plt.xticks([i for i in index], dimensions, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plot_block_{block_size}_iter_{iterations}_speedup.png')
    plt.show()

# Generate plots for each combination of block size and iteration count
plot_experiment(data, block_size=16, iterations=50)
plot_experiment(data, block_size=16, iterations=100)
plot_experiment(data, block_size=32, iterations=50)
plot_experiment(data, block_size=32, iterations=100)