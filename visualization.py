import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_file, output_dir="plots"):
    # Read the CSV file
    metrics = pd.read_csv(csv_file)

    os.makedirs(output_dir, exist_ok=True)

    # Plot Test MAE vs. Cycles
    plt.figure(figsize=(8, 6))
    plt.plot(metrics["Cycle"], metrics["Test MAE"], marker='o', linestyle='-', color='b', label="Test MAE")
    plt.title("Test MAE Across Active Learning Cycles", fontsize=14)
    plt.xlabel("Active Learning Cycle", fontsize=12)
    plt.ylabel("Test MAE", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/test_mae_vs_cycles.png")
    plt.close()

    # Plot Labeled Dataset Growth vs. Cycles
    plt.figure(figsize=(8, 6))
    plt.plot(metrics["Cycle"], metrics["Labeled Dataset Size"], marker='o', linestyle='-', color='g', label="Labeled Dataset Size")
    plt.title("Labeled Dataset Growth Across Active Learning Cycles", fontsize=14)
    plt.xlabel("Active Learning Cycle", fontsize=12)
    plt.ylabel("Labeled Dataset Size", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/dataset_growth_vs_cycles.png")
    plt.close()

    print(f"Plots saved in {output_dir}/")

if __name__ == "__main__":

    metrics_csv = "active_learning_metrics.csv"  # Path to the metrics CSV file
    plot_metrics(metrics_csv)
