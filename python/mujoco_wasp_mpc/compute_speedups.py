#!/usr/bin/env python3
"""
Script to compare performance metrics between two log files.
Computes speedup and performance loss between two files of the same format.
"""

import re
import numpy as np
from typing import List, Dict, Tuple


def parse_file(filepath: str, extract_sim_steps: bool = False) -> Tuple[List[float], List[float], List[float]]:
    """
    Parse a file and extract 'return', 'model derivatives', and optionally 'sim steps' values.

    Args:
        filepath: Path to the input file
        extract_sim_steps: Whether to extract simulation steps data

    Returns:
        Tuple of (returns_list, model_derivatives_list, sim_steps_list)
    """
    returns = []
    model_derivatives = []
    sim_steps = []

    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return [], [], []
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        return [], [], []

    # Pattern to match "return: <number>" (the actual return value)
    return_pattern = r'^\s*return:\s*([\d.]+)'  # r'improved return by planner:\s*([\d.]+)'#r'^\s*return:\s*([\d.]+)'
    # Pattern to match "model derivatives: <number>"
    derivatives_pattern = r'model derivatives:\s*([\d.]+)'
    # Pattern to match "number of sim. steps in model derivative: <number>"
    sim_steps_pattern = r'number of sim\. steps in model derivative:\s*([\d.]+)'

    # Find all matches
    return_matches = re.findall(return_pattern, content, re.MULTILINE)
    derivatives_matches = re.findall(derivatives_pattern, content)

    # Convert to floats
    returns = [float(val) for val in return_matches]
    model_derivatives = [float(val) for val in derivatives_matches]

    # Extract sim steps if requested
    if extract_sim_steps:
        sim_steps_matches = re.findall(sim_steps_pattern, content)
        sim_steps = [float(val) for val in sim_steps_matches]
        if len(sim_steps) == 0:
            print(f"Warning: No 'number of sim. steps in model derivative' values found in {filepath}")

    if len(returns) == 0:
        print(f"Warning: No 'return' values found in {filepath}")
    if len(model_derivatives) == 0:
        print(f"Warning: No 'model derivatives' values found in {filepath}")

    return returns[:], model_derivatives[:], sim_steps[:]


def compute_metrics(file1_path: str, file2_path: str) -> Dict[str, float]:
    """
    Compute speedup and performance loss between two files.

    Speedup = avg(model_derivatives_file1) / avg(model_derivatives_file2)
    Performance Loss = (avg_return_file2 - avg_return_file1) / avg_return_file1

    Args:
        file1_path: Path to the first file
        file2_path: Path to the second file

    Returns:
        Dictionary with computed metrics
    """
    print(f"Reading file 1: {file1_path}")
    returns1, derivatives1, _ = parse_file(file1_path, extract_sim_steps=False)

    print(f"Reading file 2: {file2_path}")
    returns2, derivatives2, sim_steps2 = parse_file(file2_path, extract_sim_steps=True)

    # Calculate averages
    avg_return1 = sum(returns1) / len(returns1) if returns1 else 0
    avg_return2 = sum(returns2) / len(returns2) if returns2 else 0
    avg_derivatives1 = sum(derivatives1) / len(derivatives1) if derivatives1 else 0
    avg_derivatives2 = sum(derivatives2) / len(derivatives2) if derivatives2 else 0
    avg_sim_steps2 = sum(sim_steps2) / len(sim_steps2) if sim_steps2 else 0

    # Compute metrics
    speedup = avg_derivatives1 / avg_derivatives2 if avg_derivatives2 != 0 else float('inf')
    performance_loss = avg_return1 / avg_return2 if avg_return2 != 0 else float('inf')

    # Compute MAPE for returns
    mape_returns = 0
    if returns1 and returns2:
        min_len = min(len(returns1), len(returns2))
        if min_len > 0:
            mape_returns = np.sum(np.abs(
                (np.array(returns1[:min_len]) - np.array(returns2[:min_len])))) / np.sum(
                np.abs(np.array(returns1[:min_len])))

    # Compute Normalized Root Mean Square Deviation (NRMSD) for returns
    nrmsd_returns = 0
    if returns1 and returns2:
        min_len = min(len(returns1), len(returns2))
        if min_len > 0:
            # Calculate RMSD
            rmsd = np.sqrt(np.mean((np.array(returns1[:min_len]) - np.array(returns2[:min_len])) ** 2))
            # Normalize by the range of the reference data (file1)
            data_range = np.max(returns1[:min_len]) - np.min(returns1[:min_len])
            nrmsd_returns = rmsd / data_range if data_range != 0 else 0

            # Alternative normalization options (commented out):
            # Normalize by mean of reference data:
            # nrmsd_returns = rmsd / np.mean(returns1[:min_len]) if np.mean(returns1[:min_len]) != 0 else 0
            # Normalize by standard deviation of reference data:
            # nrmsd_returns = rmsd / np.std(returns1[:min_len]) if np.std(returns1[:min_len]) != 0 else 0

    return {
        'avg_return_file1': avg_return1,
        'avg_return_file2': avg_return2,
        'avg_derivatives_file1': avg_derivatives1,
        'avg_derivatives_file2': avg_derivatives2,
        'avg_sim_steps_file2': avg_sim_steps2,
        'num_sim_steps_entries': len(sim_steps2),
        'speedup': speedup,
        'performance_loss': performance_loss,
        'num_iterations_file1': len(returns1),
        'num_iterations_file2': len(returns2),
        'mape_returns': mape_returns,
        'nrmsd_returns': nrmsd_returns
    }


def main():
    """Main function to run the comparison."""

    # ============================================================
    # CONFIGURE YOUR FILE PATHS HERE
    # ============================================================
    FILE1_PATH = "swimmer_ilqg_fd.out"  # Change this to your first file path
    FILE2_PATH = "humanoid_walk_wasp.out"  # Change this to your second file path
    # ============================================================

    print("=" * 60)
    print("Performance Metrics Comparison")
    print("=" * 60)

    metrics = compute_metrics(FILE1_PATH, FILE2_PATH)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nFile 1: {FILE1_PATH}")
    print(f"  - Number of iterations: {metrics['num_iterations_file1']}")
    print(f"  - Average return: {metrics['avg_return_file1']:.6f}")
    print(f"  - Average model derivatives: {metrics['avg_derivatives_file1']:.3f}")

    print(f"\nFile 2: {FILE2_PATH}")
    print(f"  - Number of iterations: {metrics['num_iterations_file2']}")
    print(f"  - Average return: {metrics['avg_return_file2']:.6f}")
    print(f"  - Average model derivatives: {metrics['avg_derivatives_file2']:.3f}")
    if metrics['num_sim_steps_entries'] > 0:
        print(f"  - Average sim. steps in model derivative: {metrics['avg_sim_steps_file2']:.2f}")
        print(f"    (from {metrics['num_sim_steps_entries']} entries)")

    print("\n" + "-" * 60)
    print("COMPUTED METRICS")
    print("-" * 60)

    print(f"\nSpeedup: {metrics['speedup']:.3f}x")
    if metrics['speedup'] > 1:
        print(f"  → File 2 is {metrics['speedup']:.2f}x faster than File 1")
    elif metrics['speedup'] < 1:
        print(f"  → File 2 is {1 / metrics['speedup']:.2f}x slower than File 1")
    else:
        print(f"  → Both files have the same speed")

    print(f"\nPerformance Loss: {metrics['performance_loss']:.4f} ({metrics['performance_loss'] * 100:.2f}%)")
    if metrics['performance_loss'] > 0:
        print(f"  → File 2 has {metrics['performance_loss'] * 100:.2f}% better performance than File 1")
    elif metrics['performance_loss'] < 0:
        print(f"  → File 2 has {abs(metrics['performance_loss']) * 100:.2f}% worse performance than File 1")
    else:
        print(f"  → Both files have the same performance")

    print(f"\nwMAPE (Returns): {metrics['mape_returns']:.4f}")
    print(f"  → Average percentage difference between return curves")

    print(f"\nNRMSD (Returns): {metrics['nrmsd_returns']:.4f}")
    print(f"  → Normalized root mean square deviation between return curves")
    print(f"  → Lower values indicate better agreement between the curves")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()