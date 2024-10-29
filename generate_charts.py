import pandas as pd
import matplotlib.pyplot as plt

# Read the experiment results from the CSV file
data = pd.read_csv('experiment_results.csv')

# Define a function to plot CPU vs GPU times and Speedup for each configuration
def plot_experiment(data, block_size, grid_size, iterations):
    # Filter data for the given block size, grid size, and iteration count
    filtered_data = data[(data['block_size'] == block_size) & 
                         (data['grid_size'] == grid_size) & 
                         (data['iterations'] == iterations)]
    dimensions = filtered_data['grid_dimension'].astype(str)
    
    # Define bar width and index positions
    bar_width = 0.35
    index = range(len(dimensions))
    
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
    plt.title(f'Speedup for Block Size {block_size}, Grid Size {grid_size}, {iterations} Iterations')
    plt.xticks([i for i in index], dimensions, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plot_block_{block_size}_grid_{grid_size}_iter_{iterations}_speedup.png')
    # plt.show()

# Loop through all unique combinations of block size, grid size, and iteration count
for block_size in data['block_size'].unique():
    for grid_size in data['grid_size'].unique():
        for iterations in data['iterations'].unique():
            plot_experiment(data, block_size, grid_size, iterations)