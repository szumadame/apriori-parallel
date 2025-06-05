import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apriori_parallel import (
    run_apriori
)

def benchmark_sizeup(df, itemset_col, min_support_threshold, min_confidence_threshold, processor_counts):
    total_len = len(df)
    fractions = [p / processor_counts[-1] for p in processor_counts] 
    times = []

    for n_proc, frac in zip(processor_counts, fractions):
        df_sample = df.sample(frac=frac, random_state=42).reset_index(drop=True)
        print(f"Dataset size: {len(df_sample)}, Processes: {n_proc}")

        start = time.time()
        run_apriori(df_sample, itemset_col, min_support_threshold, min_confidence_threshold, n_proc)
        end = time.time()

        elapsed = end - start
        times.append(elapsed)
        print(f"Elapsed time: {elapsed:.2f} seconds\n")

    return fractions, times

def plot_sizeup_results(processor_counts, fractions, times, min_support_threshold, min_confidence_threshold, df_name):
    plt.figure(figsize=(10, 5))
    plt.plot(processor_counts, times, 'o-', label='Execution Time')
    plt.xlabel('Number of Processes')
    plt.ylabel('Execution Time (s)')
    plt.title('Size-up Test: Execution Time vs. Processes (Data scales with processes)')
    plt.grid(True)
    plt.legend()

    filename = f"results/data_and_partition_scaleup/sizeup_{df_name}_supp={min_support_threshold}_conf={min_confidence_threshold}.png"
    plt.savefig(filename)
    print(f"Saved size-up plot to {filename}")

    plt.show()

    print("\n--- Size-up Results ---")
    print("Processes | Dataset Fraction | Time (s)")
    for p, f, t in zip(processor_counts, fractions, times):
        print(f"{p:<10} | {f:<17.2f} | {t:.2f}")

if __name__ == "__main__":
    df_name = "mushroom_apriori"
    df = pd.read_pickle(f"/workspaces/apriori-parallel/data/{df_name}_df.pkl")
    itemset_col = 'Categories'
    min_support_threshold = 0.4
    min_confidence_threshold = 0.4
    processor_counts = [1, 2, 4, 8, 16, 32]

    fractions, times = benchmark_sizeup(
        df, itemset_col, min_support_threshold, min_confidence_threshold, processor_counts
    )
    plot_sizeup_results(processor_counts, fractions, times, min_support_threshold, min_confidence_threshold, df_name)
