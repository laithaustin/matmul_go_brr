#!/usr/bin/env python3

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for SSH
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

# Set nice styling defaults
try:
    matplotlib.style.use("fivethirtyeight")
except:
    pass
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.facecolor"] = "white"

# Kernel names mapping
KERNEL_NAMES = {
    0: "cuBLAS",
    1: "Naive",
    2: "GMEM Coalescing",
    3: "SMEM Caching",
    4: "1D Blocktiling",
    5: "2D Blocktiling",
    6: "Vectorized Mem Access",
    7: "Avoid Bank Conflicts (Linearize)",
    8: "Avoid Bank Conflicts (Offset)",
    9: "Autotuning",
    10: "Warptiling",
    11: "Double Buffering",
}


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
    return KERNEL_NAMES.get(kernel_num, f"Kernel {kernel_num}")


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
    """Plot all available kernels on one graph with labels on the plot."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all data
    all_data = []
    kernels_plotted = []

    for kernel_num in range(12):  # Support up to 12 kernels
        data = load_kernel_data(kernel_num)
        if data is not None:
            for _, row in data.iterrows():
                all_data.append({
                    "kernel": kernel_num,
                    "size": row["Size"],
                    "gflops": row["GFLOPS"]
                })
            kernels_plotted.append(kernel_num)

    if not kernels_plotted:
        print("No kernel data found! Run benchmarks first.")
        return

    df = pd.DataFrame(all_data)

    # Create the plot
    plt.figure(figsize=(18, 10))
    colors = sn.color_palette("husl", len(kernels_plotted))

    # Plot lines and points
    sn.lineplot(data=df, x="size", y="gflops", hue="kernel", palette=colors)
    sn.scatterplot(data=df, x="size", y="gflops", hue="kernel", palette=colors, legend=False)

    # Set ticks at actual sizes
    plt.xticks(sorted(df["size"].unique()))
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    # Display kernel names right next to the corresponding line
    for i, kernel in enumerate(kernels_plotted):
        kernel_data = df[df["kernel"] == kernel]
        plt.text(
            kernel_data["size"].iloc[-1],
            kernel_data["gflops"].iloc[-1] + 300,
            f"{kernel}:{get_kernel_name(kernel)}",
            color=colors[i],
            horizontalalignment="left",
            weight="medium",
        )

    # Turn off the legend
    plt.gca().get_legend().remove()

    plt.title("Performance of different kernels")
    plt.xlabel("Matrix size (square, one side)")
    plt.ylabel("GFLOPs/s")
    plt.tight_layout()

    output_file = f"{output_dir}/benchmark_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    plt.close()


def print_summary():
    """Print summary of all available kernel results."""
    print("\n" + "=" * 60)
    print("SGEMM Benchmark Summary")
    print("=" * 60)

    for kernel_num in range(12):
        data = load_kernel_data(kernel_num)
        if data is not None:
            name = get_kernel_name(kernel_num)
            max_gflops = data["GFLOPS"].max()
            max_size = data.loc[data["GFLOPS"].idxmax(), "Size"]
            avg_gflops = data["GFLOPS"].mean()

            print(
                f"{name:30} - Max: {max_gflops:8.2f} GFLOPS (size {max_size:4.0f}), "
                f"Avg: {avg_gflops:8.2f} GFLOPS"
            )

    print("=" * 60 + "\n")


def update_readme_table(target_size=4096):
    """Update README.md with benchmark results table for a specific matrix size."""
    # Collect data for the target size
    results = []

    for kernel_num in range(12):
        data = load_kernel_data(kernel_num)
        if data is not None:
            # Find the row with the target size
            size_data = data[data["Size"] == target_size]
            if not size_data.empty:
                gflops = size_data["GFLOPS"].iloc[0]
                results.append({
                    "kernel": kernel_num,
                    "gflops": gflops
                })

    if not results:
        print(f"No data found for matrix size {target_size}")
        return

    # Create DataFrame and sort by performance
    df = pd.DataFrame(results).sort_values(by="gflops", ascending=True)

    # Add kernel names and relative performance
    df["kernel"] = df["kernel"].apply(lambda k: f"{k}: {get_kernel_name(k)}")

    # Calculate relative performance to cuBLAS
    cublas_row = df[df["kernel"].str.startswith("0:")]
    if not cublas_row.empty:
        cublas_gflops = cublas_row["gflops"].iloc[0]
        df["relperf"] = df["gflops"].apply(lambda x: f"{x/cublas_gflops*100:.1f}%")
    else:
        df["relperf"] = "N/A"

    # Rename columns for display
    df.columns = ["Kernel", "GFLOPs/s", "Performance relative to cuBLAS"]

    # Read current README
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        print(f"README.md not found, skipping update")
        return

    with open(readme_path, "r") as f:
        readme = f.read()

    # Update the benchmark results section
    table_md = df.to_markdown(index=False)
    new_section = f"<!-- benchmark_results -->\n{table_md}\n<!-- benchmark_results -->"

    if "<!-- benchmark_results -->" in readme:
        # Replace existing section
        updated_readme = re.sub(
            r"<!-- benchmark_results -->.*<!-- benchmark_results -->",
            new_section,
            readme,
            flags=re.DOTALL,
        )

        with open(readme_path, "w") as f:
            f.write(updated_readme)

        print(f"Updated README.md with benchmark results for size {target_size}")
    else:
        print("No <!-- benchmark_results --> markers found in README.md")
        print("Add the following markers to your README where you want the table:")
        print("<!-- benchmark_results -->")
        print("<!-- benchmark_results -->")


def main():
    if len(sys.argv) == 1:
        # No arguments - print summary, plot all, and update README
        print_summary()
        plot_all_kernels()
        update_readme_table()
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
        print("  python plot.py              # Summary + plot all kernels + update README")
        print("  python plot.py <K>          # Plot single kernel")
        print("  python plot.py <K1> <K2>    # Compare two kernels")
        print("\nExample:")
        print("  python plot.py 0 1          # Compare cuBLAS vs Kernel 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
