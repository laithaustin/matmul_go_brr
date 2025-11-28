#!/usr/bin/env python3

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for SSH
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def load_kernel_data(kernel_num):
    """Load benchmark data from test file."""
    filename = f"test/test_kernel_{kernel_num}.txt"

    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        print(f"Run './sgemm {kernel_num}' first to generate data.")
        return None

    # Read CSV file
    df = pd.read_csv(filename)

    return df


def get_kernel_name(kernel_num):
    """Get display name for kernel."""
    if kernel_num == 0:
        return "cuBLAS"
    return f"Kernel {kernel_num}"


def plot_comparison(kernel1, kernel2, output_dir="images"):
    """Plot comparison between two kernels."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data1 = load_kernel_data(kernel1)
    data2 = load_kernel_data(kernel2)

    if data1 is None or data2 is None:
        return

    # Create plot
    plt.figure(figsize=(12, 7))

    # Plot kernel 1
    name1 = get_kernel_name(kernel1)
    plt.plot(
        data1["Size"],
        data1["GFLOPS"],
        marker="s",
        markersize=6,
        linewidth=2,
        color="black",
        label=name1,
    )

    # Plot kernel 2
    name2 = get_kernel_name(kernel2)
    plt.plot(
        data2["Size"],
        data2["GFLOPS"],
        marker="^",
        markersize=6,
        linewidth=2,
        color="blue",
        label=name2,
    )

    # Formatting
    plt.xlabel("Matrix Size (M = N = K)", fontsize=14, fontweight="bold")
    plt.ylabel("Performance (GFLOPS)", fontsize=14, fontweight="bold")
    plt.title(
        f"SGEMM Performance Comparison: {name1} vs {name2}",
        fontsize=16,
        fontweight="bold",
    )
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks at 256 intervals
    max_size = max(data1["Size"].max(), data2["Size"].max())
    ticks = list(range(0, max_size + 256, 256))
    plt.xticks(ticks, rotation=45)

    plt.tight_layout()

    # Save figure
    kernel1_name = "cublas" if kernel1 == 0 else str(kernel1)
    kernel2_name = "cublas" if kernel2 == 0 else str(kernel2)
    output_file = f"{output_dir}/kernel_{kernel1_name}_vs_{kernel2_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    plt.close()


def plot_single(kernel_num, output_dir="images"):
    """Plot single kernel performance."""
    os.makedirs(output_dir, exist_ok=True)

    data = load_kernel_data(kernel_num)
    if data is None:
        return

    plt.figure(figsize=(12, 7))

    name = get_kernel_name(kernel_num)
    plt.plot(
        data["Size"],
        data["GFLOPS"],
        marker="o",
        markersize=6,
        linewidth=2,
        color="steelblue",
        label=name,
    )

    plt.xlabel("Matrix Size (M = N = K)", fontsize=14, fontweight="bold")
    plt.ylabel("Performance (GFLOPS)", fontsize=14, fontweight="bold")
    plt.title(f"SGEMM Performance: {name}", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    max_size = data["Size"].max()
    ticks = list(range(0, max_size + 256, 256))
    plt.xticks(ticks, rotation=45)

    plt.tight_layout()

    kernel_name = "cublas" if kernel_num == 0 else str(kernel_num)
    output_file = f"{output_dir}/kernel_{kernel_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    plt.close()


def plot_all_kernels(output_dir="images"):
    """Plot all available kernels on one graph."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(14, 8))

    colors = ["black", "red", "blue", "green", "orange", "purple", "brown", "pink"]
    markers = ["o", "s", "^", "v", "D", "*", "p", "h"]

    kernels_plotted = []

    for kernel_num in range(8):  # 0-7
        data = load_kernel_data(kernel_num)
        if data is not None:
            name = get_kernel_name(kernel_num)
            plt.plot(
                data["Size"],
                data["GFLOPS"],
                marker=markers[kernel_num % len(markers)],
                markersize=6,
                linewidth=2,
                color=colors[kernel_num % len(colors)],
                label=name,
            )
            kernels_plotted.append(kernel_num)

    if not kernels_plotted:
        print("No kernel data found! Run benchmarks first.")
        return

    plt.xlabel("Matrix Size (M = N = K)", fontsize=14, fontweight="bold")
    plt.ylabel("Performance (GFLOPS)", fontsize=14, fontweight="bold")
    plt.title(
        "SGEMM Performance Comparison - All Kernels", fontsize=16, fontweight="bold"
    )
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = f"{output_dir}/all_kernels.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    plt.close()


def print_summary():
    """Print summary of all available kernel results."""
    print("\n" + "=" * 60)
    print("SGEMM Benchmark Summary")
    print("=" * 60)

    for kernel_num in range(8):
        data = load_kernel_data(kernel_num)
        if data is not None:
            name = get_kernel_name(kernel_num)
            max_gflops = data["GFLOPS"].max()
            max_size = data.loc[data["GFLOPS"].idxmax(), "Size"]
            avg_gflops = data["GFLOPS"].mean()

            print(
                f"{name:15} - Max: {max_gflops:8.2f} GFLOPS (size {max_size:4.0f}), "
                f"Avg: {avg_gflops:8.2f} GFLOPS"
            )

    print("=" * 60 + "\n")


def main():
    if len(sys.argv) == 1:
        # No arguments - print summary and plot all
        print_summary()
        plot_all_kernels()
    elif len(sys.argv) == 2:
        # Single kernel plot
        kernel = int(sys.argv[1])
        plot_single(kernel)
    elif len(sys.argv) == 3:
        # Compare two kernels
        kernel1 = int(sys.argv[1])
        kernel2 = int(sys.argv[2])
        plot_comparison(kernel1, kernel2)
    else:
        print("Usage:")
        print("  python plot.py              # Summary + plot all kernels")
        print("  python plot.py <K>          # Plot single kernel")
        print("  python plot.py <K1> <K2>    # Compare two kernels")
        print("\nExample:")
        print("  python plot.py 0 1          # Compare cuBLAS vs Kernel 1")
        sys.exit(1)

        print("\nExample:")
        print("  python plot.py 0 1          # Compare cuBLAS vs Kernel 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
