import re
import matplotlib.pyplot as plt
import numpy as np
import os


def parse_data(file_path):
    """
    Parse the data file and extract return values and model derivative times.

    Args:
        file_path (str): Path to the data file

    Returns:
        tuple: (returns, model_derivatives) - lists of parsed values
    """
    returns = []
    model_derivatives = []

    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # Regular expressions to match the patterns
        return_pattern = r'return:\s+([\d.]+)'
        model_derivative_pattern = r'model derivative \(ms\):\s+([\d.]+)'

        # Find all matches
        return_matches = re.findall(return_pattern, content)
        derivative_matches = re.findall(model_derivative_pattern, content)

        # Convert to float
        returns = [float(x) for x in return_matches]
        model_derivatives = [float(x) for x in derivative_matches]

        print(f"Parsed {len(returns)} return values and {len(model_derivatives)} derivative values from {file_path}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return [], []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return [], []

    return returns, model_derivatives


def create_comparison_plots(file1_data, file2_data, file1_name, file2_name, save_plots=False):
    """
    Create comparison plots for returns and model derivatives from two files.

    Args:
        file1_data (tuple): (returns, model_derivatives) from first file
        file2_data (tuple): (returns, model_derivatives) from second file
        file1_name (str): Name/label for first file
        file2_name (str): Name/label for second file
        save_plots (bool): Whether to save plots to files
    """
    returns1, derivatives1 = file1_data
    returns2, derivatives2 = file2_data

    if not (returns1 and derivatives1 and returns2 and derivatives2):
        print("Insufficient data to create comparison plots.")
        return

    # Create iteration indices
    iterations1_returns = list(range(1, len(returns1) + 1))
    iterations2_returns = list(range(1, len(returns2) + 1))
    iterations1_derivatives = list(range(1, len(derivatives1) + 1))
    iterations2_derivatives = list(range(1, len(derivatives2) + 1))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Plot 1: Returns Comparison
    ax1.plot(iterations1_returns, returns1, 'b-', linewidth=1.5, alpha=0.8, label=file1_name)
    ax1.plot(iterations2_returns, returns2, 'r-', linewidth=1.5, alpha=0.8, label=file2_name)
    ax1.set_title('Return Values Comparison Over Iterations', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Return', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Add mean lines for returns
    mean_return1 = np.mean(returns1)
    mean_return2 = np.mean(returns2)
    ax1.axhline(y=mean_return1, color='b', linestyle='--', alpha=0.6,
                label=f'{file1_name} Mean: {mean_return1:.4f}')
    ax1.axhline(y=mean_return2, color='r', linestyle='--', alpha=0.6,
                label=f'{file2_name} Mean: {mean_return2:.4f}')
    ax1.legend(fontsize=10)

    # Set x-axis limits for returns plot
    max_iterations_returns = max(len(returns1), len(returns2))
    ax1.set_xlim(1, max_iterations_returns)

    # Plot 2: Model Derivatives Comparison
    ax2.plot(iterations1_derivatives, derivatives1, 'g-', linewidth=1.5, alpha=0.8, label=file1_name)
    ax2.plot(iterations2_derivatives, derivatives2, 'm-', linewidth=1.5, alpha=0.8, label=file2_name)
    ax2.set_title('Model Derivative Time Comparison Over Iterations', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Model Derivative (ms)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    # Add mean lines for derivatives
    mean_derivative1 = np.mean(derivatives1)
    mean_derivative2 = np.mean(derivatives2)
    ax2.axhline(y=mean_derivative1, color='g', linestyle='--', alpha=0.6,
                label=f'{file1_name} Mean: {mean_derivative1:.2f} ms')
    ax2.axhline(y=mean_derivative2, color='m', linestyle='--', alpha=0.6,
                label=f'{file2_name} Mean: {mean_derivative2:.2f} ms')
    ax2.legend(fontsize=10)

    # Set x-axis limits for derivatives plot
    max_iterations_derivatives = max(len(derivatives1), len(derivatives2))
    ax2.set_xlim(1, max_iterations_derivatives)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save plots if requested
    if save_plots:
        plt.savefig('comparison_plots6'
                    '.png', dpi=300, bbox_inches='tight')
        print("Comparison plots saved as 'comparison_plots.png'")

    # Show the plots
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\n{file1_name.upper()} - RETURNS:")
    print(f"  Count: {len(returns1)}")
    print(f"  Mean: {mean_return1:.6f}")
    print(f"  Std: {np.std(returns1):.6f}")
    print(f"  Min: {min(returns1):.6f}")
    print(f"  Max: {max(returns1):.6f}")

    print(f"\n{file2_name.upper()} - RETURNS:")
    print(f"  Count: {len(returns2)}")
    print(f"  Mean: {mean_return2:.6f}")
    print(f"  Std: {np.std(returns2):.6f}")
    print(f"  Min: {min(returns2):.6f}")
    print(f"  Max: {max(returns2):.6f}")

    print(f"\n{file1_name.upper()} - MODEL DERIVATIVE (ms):")
    print(f"  Count: {len(derivatives1)}")
    print(f"  Mean: {mean_derivative1:.2f}")
    print(f"  Std: {np.std(derivatives1):.2f}")
    print(f"  Min: {min(derivatives1):.2f}")
    print(f"  Max: {max(derivatives1):.2f}")

    print(f"\n{file2_name.upper()} - MODEL DERIVATIVE (ms):")
    print(f"  Count: {len(derivatives2)}")
    print(f"  Mean: {mean_derivative2:.2f}")
    print(f"  Std: {np.std(derivatives2):.2f}")
    print(f"  Min: {min(derivatives2):.2f}")
    print(f"  Max: {max(derivatives2):.2f}")

    # Print comparison insights
    print("\n" + "=" * 60)
    print("COMPARISON INSIGHTS")
    print("=" * 60)

    return_diff = mean_return2 - mean_return1
    derivative_diff = mean_derivative2 - mean_derivative1

    print(f"\nReturn Difference ({file2_name} - {file1_name}): {return_diff:+.6f}")
    if abs(return_diff) > 0.001:
        better_return = file2_name if return_diff > 0 else file1_name
        print(f"  → {better_return} has better average returns")
    else:
        print("  → Returns are very similar")

    print(f"\nDerivative Time Difference ({file2_name} - {file1_name}): {derivative_diff:+.2f} ms")
    if abs(derivative_diff) > 5:
        faster_method = file1_name if derivative_diff > 0 else file2_name
        print(f"  → {faster_method} is faster on average")
    else:
        print("  → Computation times are similar")


def get_file_display_name(file_path):
    """
    Extract a display name from file path (without extension).
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def main():
    """
    Main function to run the data processing and comparison plotting.
    """
    # Specify the paths to your data files
    file1_path = 'quadruped_static_walk.fd.out'  # Change this to your first file path
    file2_path = 'test.xml.out'  # Change this to your second file path

    # Get display names for the files
    file1_name = get_file_display_name(file1_path)
    file2_name = get_file_display_name(file2_path)

    print("Parsing data files...")

    # Parse both files
    file1_data = parse_data(file1_path)
    file2_data = parse_data(file2_path)

    # Check if both files were parsed successfully
    if (file1_data[0] and file1_data[1] and file2_data[0] and file2_data[1]):
        print("Creating comparison plots...")
        create_comparison_plots(file1_data, file2_data, file1_name, file2_name, save_plots=True)
    else:
        print("Failed to parse one or both data files. Please check your file paths and formats.")
        if not (file1_data[0] and file1_data[1]):
            print(f"  - Issue with {file1_path}")
        if not (file2_data[0] and file2_data[1]):
            print(f"  - Issue with {file2_path}")


if __name__ == "__main__":
    main()