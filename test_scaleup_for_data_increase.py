import pandas as pd
import numpy as np
import time
import os
import psutil
import matplotlib.pyplot as plt
from apriori_parallel import (
    run_apriori
)

def measure_time_and_memory(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start


def test_scaling_by_dataset_size(df, itemset_col, min_support, min_conf, n_partitions):
    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    times = []
    for frac in fractions:
        subset_df = df.sample(frac=frac, random_state=42).reset_index(drop=True)
        print(f"Testing with {len(subset_df)} transactions ({frac*100:.0f}%)")
        _, t = measure_time_and_memory(
            run_apriori, subset_df, itemset_col, min_support, min_conf, n_partitions
        )
        times.append(t)
        print(f"Time: {t:.2f}s")
    return fractions, times


def plot_results(x, times, xlabel, title_suffix, filename_prefix, supp, conf):
    plt.figure(figsize=(10, 5))

    plt.plot(x, times, marker='o', color='b')
    plt.xlabel(xlabel)
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time vs {title_suffix}')

    plt.tight_layout()
    
    if not os.path.exists("results"):
        os.makedirs("results")
    filename = f"results/data_scaleup/{filename_prefix}_supp={supp}_conf={conf}.png"
    plt.savefig(filename)
    print(f"Saved plot as {filename}")

    plt.show()


if __name__ == "__main__":
    df_name = "chicago_crime"
    df = pd.read_pickle(f"/workspaces/apriori-parallel/data/{df_name}_df.pkl")
    itemset_col = 'Categories'
    min_support_threshold = 0.015
    min_confidence_threshold = 0.4

    num_runs = 5
    all_times = []

    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        fractions, times = test_scaling_by_dataset_size(
            df, itemset_col, min_support_threshold, min_confidence_threshold, n_partitions=8
        )
        all_times.append(times)

    avg_times = np.mean(all_times, axis=0).tolist()

    plot_results(
        [f"{int(f*100)}%" for f in fractions], avg_times,
        xlabel='Dataset Size', title_suffix='Dataset Size',
        filename_prefix=df_name,
        supp=min_support_threshold,
        conf=min_confidence_threshold
    )

    print("\n--- Average Execution Times (s) ---")
    for f, t in zip(fractions, avg_times):
        print(f"{int(f*100)}%: {t:.2f} seconds")
