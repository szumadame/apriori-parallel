import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apriori_parallel import (
    calculate_support_parallel,
    prune,
    generate_candidates,
    generate_association_rules
)

def run_apriori_on_df(df_chunk, itemset_col, min_support_threshold, min_confidence_threshold, n_partitions):
    all_items = set()
    for categories in df_chunk[itemset_col]:
        all_items.update(categories.split(", "))

    all_supports = {}
    size = 1

    pruned_support = prune(
        calculate_support_parallel(df_chunk, itemset_col, [(item,) for item in all_items], n_partitions),
        min_support_threshold
    )
    all_supports.update(pruned_support)

    while pruned_support:
        size += 1
        candidate_itemsets = generate_candidates(pruned_support, size)
        if not candidate_itemsets:
            break

        support_itemsets = calculate_support_parallel(df_chunk, itemset_col, candidate_itemsets, n_partitions)
        pruned_support = prune(support_itemsets, min_support_threshold)

        if pruned_support:
            all_supports.update(pruned_support)
        else:
            break

    rules = generate_association_rules(all_supports, min_confidence_threshold)
    return all_supports, rules

def benchmark_weak_scaling(df, itemset_col, min_support_threshold, min_confidence_threshold, processor_counts):
    chunks = np.array_split(df, 32)
    response_times = []

    for n_proc in processor_counts:
        # Use the first n_proc chunks for the test
        df_for_test = pd.concat(chunks[:n_proc])
        print(f"Running weak scaling with {n_proc} processors on {len(df_for_test)} transactions...")

        start = time.time()
        # Each processor processes its 1/32-sized chunk in parallel (using calculate_support_parallel internally!)
        run_apriori_on_df(df_for_test, itemset_col, min_support_threshold, min_confidence_threshold, n_proc)
        end = time.time()

        elapsed = end - start
        response_times.append(elapsed)
        print(f"Elapsed time: {elapsed:.2f} seconds\n")

    return response_times

def plot_normalized_times(processor_counts, times):
    baseline_time = times[0]
    normalized_times = [t / baseline_time for t in times]

    plt.figure(figsize=(10, 5))
    plt.plot(processor_counts, normalized_times, 'o-', label='Normalized Response Time')
    plt.xlabel('Number of processors')
    plt.ylabel('Normalized Response Time')
    plt.title('Weak Scaling: Normalized Response Time vs. Number of Processors')
    plt.savefig(f"results/scaleup_plot_chicago_crime_supp={min_support_threshold}conf={min_confidence_threshold}.png")

    plt.grid(True)
    plt.legend()
    plt.show()

    print("Processors | Time (s) | Normalized Time")
    for p, t, norm in zip(processor_counts, times, normalized_times):
        print(f"{p:<10} | {t:<10.2f} | {norm:.2f}")

if __name__ == "__main__":
    df = pd.read_pickle("/workspaces/apriori-parallel/data/chicago_crime_df.pkl")
    itemset_col = 'Categories'
    min_support_threshold = 0.01
    min_confidence_threshold = 0.4
    processor_counts = [1, 2, 4, 6, 8, 10, 12, 24, 32]

    times = benchmark_weak_scaling(
        df, itemset_col, min_support_threshold, min_confidence_threshold, processor_counts
    )
    plot_normalized_times(processor_counts, times)
