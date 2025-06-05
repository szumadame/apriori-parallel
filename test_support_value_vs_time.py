import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from apriori_parallel import (
    run_apriori
)

def measure_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start


def test_scaling_by_support(df, itemset_col, min_conf, n_partitions):
    supports = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    times = []
    for supp in supports:
        print(f"Testing with min_support={supp:.3f}")
        _, t = measure_time(
            run_apriori, df, itemset_col, supp, min_conf, n_partitions
        )
        times.append(t)
        print(f"Time: {t:.2f}s")
    return supports, times


def plot_results(x, times, xlabel, title_suffix, filename_prefix, conf):
    plt.figure(figsize=(10, 5))

    plt.plot(x, times, marker='o', color='b')
    plt.xlabel(xlabel)
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time vs {title_suffix}')

    plt.tight_layout()
    
    if not os.path.exists("results"):
        os.makedirs("results")
    filename = f"results/support_size/{filename_prefix}_conf={conf}.png"
    plt.savefig(filename)
    print(f"Saved plot as {filename}")

    plt.show()


if __name__ == "__main__":
    df_name = "mushroom_apriori"
    df = pd.read_pickle(f"/workspaces/apriori-parallel/data/{df_name}_df.pkl")
    itemset_col = 'Categories'
    min_confidence_threshold = 0.4

    num_runs = 5
    all_times = []

    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        supports, times = test_scaling_by_support(
            df, itemset_col, min_confidence_threshold, n_partitions=8
        )
        all_times.append(times)

    avg_times = np.mean(all_times, axis=0).tolist()

    plot_results(
        supports, avg_times,
        xlabel='Min Support Threshold', title_suffix='Min Support',
        filename_prefix=df_name,
        conf=min_confidence_threshold
    )

    print("\n--- Average Execution Times (s) ---")
    for s, t in zip(supports, avg_times):
        print(f"Support={s:.3f}: {t:.2f} seconds")
