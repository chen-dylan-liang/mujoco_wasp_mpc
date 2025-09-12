import re
import matplotlib.pyplot as plt
import numpy as np


def parse_timing_data(filename):
    """
    Parse the timing data from the file and extract the four timing components.
    """
    model_derivatives = []
    cost_derivatives = []
    solver_steps = []
    line_searches = []

    with open(filename, 'r') as file:
        content = file.read()

    # Regular expressions to match each timing component
    model_pattern = r'model derivatives:\s+([\d.]+)'
    cost_pattern = r'cost derivatives:\s+([\d.]+)'
    solver_pattern = r'solver step:\s+([\d.]+)'
    line_pattern = r'line search:\s+([\d.]+)'

    # Extract all values for each component
    model_derivatives = [float(x) for x in re.findall(model_pattern, content)]
    cost_derivatives = [float(x) for x in re.findall(cost_pattern, content)]
    solver_steps = [float(x) for x in re.findall(solver_pattern, content)]
    line_searches = [float(x) for x in re.findall(line_pattern, content)]

    return model_derivatives, cost_derivatives, solver_steps, line_searches


def create_single_pie_chart(model_derivatives, cost_derivatives, solver_steps, line_searches, ax, title=""):
    """
    Create a single pie chart on the given axis.
    """
    # Calculate averages
    avg_model = np.mean(model_derivatives)
    avg_cost = np.mean(cost_derivatives)
    avg_solver = np.mean(solver_steps)
    avg_line = np.mean(line_searches)

    # Calculate total and percentages
    total_time = avg_model + avg_cost + avg_solver + avg_line
    percentages = [
        (avg_model / total_time) * 100,
        (avg_cost / total_time) * 100,
        (avg_solver / total_time) * 100,
        (avg_line / total_time) * 100
    ]

    # Labels and contrasting colors suitable for academic papers
    labels = ['Model Derivatives', 'Cost Derivatives', 'Solver Step', 'Line Search']
    colors = ['#27AE60', '#E74C3C', '#F39C12', '#2C3E50']  # Green, red, orange, dark blue

    # Create the pie chart on the given axis with larger percentage text
    wedges, texts, autotexts = ax.pie(percentages, colors=colors,
                                      autopct='%1.1f%%', startangle=90)

    # Make percentage text larger and more readable for papers
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)  # Increased from 10 to 14

    # Add title with larger font
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)  # Increased from 12 to 16

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    return wedges, labels, percentages, total_time


def create_four_pie_charts():
    """
    Create a 2x2 grid of pie charts from four different files with better spacing for papers.
    """
    filenames = [
        'pie_chart_ilqg_fd.out',
        'pie_chart_gd_fd.out',
        'pie_chart_ilqg_wasp.out',
        'pie_chart_gd_wasp.out',
    ]

    titles = [
        'iLQG + FD',
        'GD + FD',
        'iLQG + WASP',
        'GD + WASP'
    ]

    # Create a 2x2 subplot with reduced spacing and better size for papers
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Reduced from 16x12 to 12x10
    axes = axes.flatten()

    all_wedges = None
    all_labels = None

    for i, (filename, title) in enumerate(zip(filenames, titles)):
        try:
            # Parse data for this file
            model_derivatives, cost_derivatives, solver_steps, line_searches = parse_timing_data(filename)

            if not model_derivatives:
                print(f"No data found in {filename}. Skipping this subplot.")
                continue

            # Create pie chart on the current axis
            wedges, labels, percentages, total_time = create_single_pie_chart(
                model_derivatives, cost_derivatives, solver_steps, line_searches,
                axes[i], title
            )

            # Store wedges and labels for legend (same for all charts)
            if all_wedges is None:
                all_wedges = wedges
                all_labels = labels

            # Print summary for this chart
            print(f"\n{title} - Summary Statistics (averaged over {len(model_derivatives)} iterations):")
            print(f"Model Derivatives: {np.mean(model_derivatives):.3f} ({percentages[0]:.1f}%)")
            print(f"Cost Derivatives:  {np.mean(cost_derivatives):.3f} ({percentages[1]:.1f}%)")
            print(f"Solver Step:       {np.mean(solver_steps):.3f} ({percentages[2]:.1f}%)")
            print(f"Line Search:       {np.mean(line_searches):.3f} ({percentages[3]:.1f}%)")
            print(f"Total Average:     {total_time:.3f}")

        except FileNotFoundError:
            print(f"File '{filename}' not found. Skipping this subplot.")
            axes[i].text(0.5, 0.5, f'File not found:\n{filename}',
                         ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
            continue
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Add a large legend in the center of the four charts
    if all_wedges is not None:
        fig.legend(all_wedges, all_labels, title="Components",
                   loc="center", bbox_to_anchor=(0.5, 0.5),
                   frameon=True, facecolor='white', edgecolor='black',
                   fontsize=18, title_fontsize=20,
                   bbox_transform=fig.transFigure)  # Much larger font sizes

    # Adjust spacing to make room for center legend
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Increased spacing to accommodate center legend

    return fig


def main():
    try:
        # Set matplotlib parameters for better paper quality
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12
        })

        # Create the four pie charts
        fig = create_four_pie_charts()

        # Show the plot
        plt.show()

        # Save the chart as PDF for paper use with higher DPI
        fig.savefig('optimization_timing_four_pie_charts.pdf', dpi=600, bbox_inches='tight')
        print("\nFour pie charts saved as 'optimization_timing_four_pie_charts.pdf'")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()