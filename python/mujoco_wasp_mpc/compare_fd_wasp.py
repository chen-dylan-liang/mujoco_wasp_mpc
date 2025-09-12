import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import matplotlib as mpl
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
plt.rcParams['axes.labelweight'] = 'bold'



def parse_file(filepath):
    """Parse a single output file and extract relevant metrics."""
    data = {
        'return': [],
        'model_derivatives': [],
        'sim_steps': []
    }

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()

            # Extract return values
            if line.startswith('return:'):
                value = float(line.split(':')[1].strip())
                data['return'].append(value)

            # Extract model derivatives (in milliseconds)
            elif line.startswith('model derivatives:'):
                value = float(line.split(':')[1].strip())
                data['model_derivatives'].append(value)

            # Extract number of simulation steps
            elif line.startswith('number of sim. steps in model derivative:'):
                value = int(line.split(':')[1].strip())
                data['sim_steps'].append(value)

    except FileNotFoundError:
        print(f"Warning: File {filepath} not found")
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    data['return'] = data['return'][100:]
    data['model_derivatives'] = data['model_derivatives'][100:]
    data['sim_steps'] = data['sim_steps'][100:]

    return data


def plot_data(file_paths, step_size=2):
    """Create plots comparing metrics from multiple files.

    Args:
        file_paths: List of file paths to plot
        step_size: Plot every Nth point (default=2 for every other point)
    """

    # Parse all files
    all_data = {}
    legends = []

    for filepath in file_paths:
        # Create legend name by removing .out extension
        legend_name = filepath.replace('.out', '')
        legends.append(legend_name)
        all_data[legend_name] = parse_file(filepath)
        if len(all_data[legend_name]['sim_steps'])==0:
            for i in range(len( all_data[legend_name]['return'])):
                all_data[legend_name]['sim_steps'].append(2487)

    # Create figure with 3 subplots - long and narrow for paper
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)  # sharex=True to share x-axis

    # Define colors for each file
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    # Plot 1: Return values (now labeled as Cost)
    ax1 = axes[0]
    for i, (legend_name, data) in enumerate(all_data.items()):
        if data['return']:
            # Plot every Nth point based on step_size
            iterations = range(1, len(data['return']) + 1, step_size)
            values = data['return'][::step_size]
            if i >3:
                ax1.plot(iterations, values,
                     label=legend_name, color=colors[i % len(colors)], linestyle=':',
                     linewidth=2, alpha=0.8)
            else:
                ax1.plot(iterations, values,
                         label=legend_name, color=colors[i % len(colors)],
                         linewidth=2, alpha=0.8)

    ax1.set_ylabel('Cost')
    ax1.grid(True, alpha=0.3)
    # Set at least 4 ticks on y-axis
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax1.tick_params(axis='y', which='major', width=1.5, length=6)
    # Make y-axis tick labels bold
    for label in ax1.get_yticklabels():
        label.set_weight('bold')

    # Plot 2: Model derivatives (in milliseconds)
    ax2 = axes[1]
    for i, (legend_name, data) in enumerate(all_data.items()):
        if data['model_derivatives']:
            # Plot every Nth point based on step_size
            iterations = range(1, len(data['model_derivatives']) + 1, step_size)
            values = data['model_derivatives'][::step_size]
            if i >3:
                ax2.plot(iterations, values,
                         label=legend_name, color=colors[i % len(colors)], linestyle=':',
                         linewidth=2, alpha=0.8)
            else:
                ax2.plot(iterations, values,
                         label=legend_name, color=colors[i % len(colors)],
                         linewidth=2, alpha=0.8)

    ax2.set_ylabel('Time (ms)')
    ax2.grid(True, alpha=0.3)
    # Set y-axis to start from 0
    y_min, y_max = ax2.get_ylim()
    ax2.set_ylim(0, y_max)
    # Set at least 4 ticks on y-axis including 0
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4, prune='upper'))
    ax2.tick_params(axis='y', which='major', width=1.5, length=6)
    # Make y-axis tick labels bold
    for label in ax2.get_yticklabels():
        label.set_weight('bold')

    # Plot 3: Number of simulation steps
    ax3 = axes[2]
    for i, (legend_name, data) in enumerate(all_data.items()):
        if data['sim_steps']:  # Only plot if data exists (FD.out won't have this)
            # Plot every Nth point based on step_size
            iterations = range(1, len(data['sim_steps']) + 1, step_size)
            values = data['sim_steps'][::step_size]
            if i >3:
                ax3.plot(iterations, values,
                         label=legend_name, color=colors[i % len(colors)], linestyle=':',
                         linewidth=2, alpha=0.8)
            else:
                ax3.plot(iterations, values,
                         label=legend_name, color=colors[i % len(colors)],
                         linewidth=2, alpha=0.8)

    ax3.set_xlabel('Iteration', )
    ax3.set_ylabel('# Sim. Steps')
    ax3.grid(True, alpha=0.3)
    # Set x-axis limits starting from 0
    ax3.set_xlim(0, 1000)
    # Force specific x-axis ticks including 0
    ax3.set_xticks([0, 200, 400, 600, 800, 1000])
    # Set at least 4 ticks on y-axis
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4, integer=True))
    ax3.tick_params(axis='both', which='major', width=1.5, length=6)
    # Make tick labels bold
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_weight('bold')

    # Add a single horizontal legend at the bottom of the figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=len(labels), prop={'weight': 'bold'})

    # Adjust layout to prevent overlap and make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the bottom legend

    # Save the figure as PDF for paper-ready output
    plt.savefig('teaser.pdf', dpi=300, bbox_inches='tight', format='pdf')
    print("\nPlot saved as 'parameter_tuning.pdf'")

    # Show the plot
    plt.show()

    # Print summary statistics
    print(f"\n=== Summary Statistics (plotting every {step_size} iterations) ===\n")
    for legend_name, data in all_data.items():
        print(f"\n{legend_name}:")
        if data['return']:
            print(f"  Cost - Min: {min(data['return']):.6f}, "
                  f"Max: {max(data['return']):.6f}, "
                  f"Mean: {np.mean(data['return']):.6f}, "
                  f"Final: {data['return'][-1]:.6f}")
        if data['model_derivatives']:
            print(f"  Model Derivatives (ms) - Mean: {np.mean(data['model_derivatives']):.3f}, "
                  f"Std: {np.std(data['model_derivatives']):.3f}")
        if data['sim_steps']:
            print(f"  Sim Steps - Mean: {np.mean(data['sim_steps']):.1f}, "
                  f"Std: {np.std(data['sim_steps']):.1f}")

    # Print overall averages across all iterations
    print("\n=== Overall Averages Across All Iterations ===\n")
    for legend_name, data in all_data.items():
        print(f"\n{legend_name}:")
        if data['return']:
            print(f"  Average Cost: {np.mean(data['return']):.6f}")
        if data['model_derivatives']:
            print(f"  Average Model Derivatives (ms): {np.mean(data['model_derivatives']):.3f}")
        if data['sim_steps']:
            print(f"  Average Sim Steps: {np.mean(data['sim_steps']):.1f}")


# Main execution
if __name__ == "__main__":
    # Define file paths
    file_paths = [
       'climb_fd.out',
        'climb_wasp.out',
    ]

    # Create plots with customizable step size
    step_size = 2 # Change this value: 1=all points, 2=every other, 5=every 5th, 10=every 10th, etc.
    plot_data(file_paths, step_size=step_size)