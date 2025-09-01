import re
import matplotlib.pyplot as plt
import numpy as np


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

        print(f"Parsed {len(returns)} return values and {len(model_derivatives)} derivative values")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return [], []
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], []

    return returns, model_derivatives


def create_plots(returns, model_derivatives, save_plots=False):
    """
    Create two separate plots for returns and model derivatives.

    Args:
        returns (list): List of return values
        model_derivatives (list): List of model derivative values
        save_plots (bool): Whether to save plots to files
    """
    if not returns or not model_derivatives:
        print("No data to plot.")
        return

    # Create iteration indices
    iterations_returns = list(range(1, len(returns) + 1))
    iterations_derivatives = list(range(1, len(model_derivatives) + 1))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Returns
    ax1.plot(iterations_returns, returns, 'b-', linewidth=1, alpha=0.7)
    ax1.set_title('Return Values Over Iterations', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Return')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(returns))

    # Add some statistics to the return plot
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    ax1.axhline(y=mean_return, color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {mean_return:.4f}')
    ax1.legend()

    # Plot 2: Model Derivatives
    ax2.plot(iterations_derivatives, model_derivatives, 'g-', linewidth=1, alpha=0.7)
    ax2.set_title('Model Derivative Time Over Iterations', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Model Derivative (ms)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(model_derivatives))

    # Add some statistics to the derivative plot
    mean_derivative = np.mean(model_derivatives)
    std_derivative = np.std(model_derivatives)
    ax2.axhline(y=mean_derivative, color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {mean_derivative:.2f} ms')
    ax2.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save plots if requested
    if save_plots:
        plt.savefig('training_plots.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'training_plots.png'")

    # Show the plots
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Returns:")
    print(f"  Count: {len(returns)}")
    print(f"  Mean: {mean_return:.6f}")
    print(f"  Std: {std_return:.6f}")
    print(f"  Min: {min(returns):.6f}")
    print(f"  Max: {max(returns):.6f}")

    print(f"\nModel Derivative (ms):")
    print(f"  Count: {len(model_derivatives)}")
    print(f"  Mean: {mean_derivative:.2f}")
    print(f"  Std: {std_derivative:.2f}")
    print(f"  Min: {min(model_derivatives):.2f}")
    print(f"  Max: {max(model_derivatives):.2f}")


def main():
    file_path = 'quadruped_static_walking.fd.out'

    print("Parsing data file...")
    returns, model_derivatives = parse_data(file_path)
    print(returns[:20])
    print(model_derivatives[:20])

    if returns and model_derivatives:
        print("Creating plots...")
        create_plots(returns, model_derivatives, save_plots=True)
    else:
        print("Failed to parse data. Please check your file path and format.")


if __name__ == "__main__":
    main()