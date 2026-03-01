import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace

# Set parameters
epsilon = 1
sum_x = 5
sum_x_prime = 6

# Create range for plotting
x_range = np.linspace(-2, 12, 1000)

scale = 1/epsilon
dist_x = laplace.pdf(x_range, loc=sum_x, scale=scale)
dist_x_prime = laplace.pdf(x_range, loc=sum_x_prime, scale=scale)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_range, dist_x, 'b-', label=f"Output distribution for x (sum={sum_x})", linewidth=2)
plt.plot(x_range, dist_x_prime, 'r-', label=f"Output distribution for x' (sum={sum_x_prime})", linewidth=2)

# Add vertical lines at the means
plt.axvline(x=sum_x, color='b', linestyle='--', alpha=0.5, label=f"Mean for x = {sum_x}")
plt.axvline(x=sum_x_prime, color='r', linestyle='--', alpha=0.5, label=f"Mean for x' = {sum_x_prime}")

# Shade the overlapping area
y_min = np.minimum(dist_x, dist_x_prime)
plt.fill_between(x_range, 0, y_min, alpha=0.3, color='gray', label='Overlapping area')

# Labels and formatting
plt.xlabel('Output Value', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title(f'Differential Privacy: Output Distributions with Laplace Noise (ε = {epsilon})', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Add text annotations
plt.text(8.5, 0.3, f'Laplace(1/ε) noise\nε = {epsilon}\nscale = {scale}',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
plt.tight_layout()
plt.savefig("laplace.pdf")
plt.show()