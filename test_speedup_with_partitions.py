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

def run_apriori(df, itemset_col, min_support_threshold, min_confidence_threshold, n_partitions):
    all_items = set()
    for categories in df[itemset_col]:
        all_items.update(categories.split(", "))

    all_supports = {}
    size = 1

    pruned_support = prune(
        calculate_support_parallel(df, itemset_col, [(item,) for item in all_items], n_partitions),
        min_support_threshold
    )
    all_supports.update(pruned_support)

    while pruned_support:
        size += 1
        candidate_itemsets = generate_candidates(pruned_support, size)
        if not candidate_itemsets:
            break

        support_itemsets = calculate_support_parallel(df, itemset_col, candidate_itemsets, n_partitions)
        pruned_support = prune(support_itemsets, min_support_threshold)

        if pruned_support:
            all_supports.update(pruned_support)
        else:
            break

    rules = generate_association_rules(all_supports, min_confidence_threshold)
    return all_supports, rules

def benchmark(df, itemset_col, min_support_threshold, min_confidence_threshold, n_partitions_list):
    times = []
    for n in n_partitions_list:
        print(f"Running with {n} partitions...")
        start = time.time()
        run_apriori(df, itemset_col, min_support_threshold, min_confidence_threshold, n)
        end = time.time()
        elapsed = end - start
        print(f"Time: {elapsed:.3f} seconds")
        times.append(elapsed)
    return times

def plot_speedup(n_partitions_list, times):
    baseline_time = times[0]
    speedups = [baseline_time / t for t in times]

    plt.figure(figsize=(10, 5))
    plt.plot(n_partitions_list, times, 'o-', label='Execution Time (s)')
    plt.xlabel('Number of processors')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs. Number of Processors')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(n_partitions_list, speedups, 'o-', label='Speedup')
    plt.xlabel('Number of processors')
    plt.ylabel('Speedup')
    plt.title('Speedup vs. Number of Processors')
    plt.grid(True)
    plt.savefig(f"results/speedup_plot_grocery_supp={min_support_threshold}conf={min_confidence_threshold}.png")
    plt.legend()
    plt.show()

    return speedups

if __name__ == "__main__":
    # Load dataset
    df = pd.read_pickle("/workspaces/apriori-parallel/data/grocery_df.pkl")
    itemset_col = 'Categories'
    min_support_threshold = 0.3
    min_confidence_threshold = 0.5

    # Define number of processors to test
    n_partitions_list = [1, 2, 4, 8, 16, 32]

    # Benchmark
    times = benchmark(df, itemset_col, min_support_threshold, min_confidence_threshold, n_partitions_list)

    # Plot and calculate speedup
    speedups = plot_speedup(n_partitions_list, times)

    # Show relative speedup (speedup relative to 1-core speedup)
    print("Number of Processors | Execution Time (s) | Speedup")
    for n, t, s in zip(n_partitions_list, times, speedups):
        print(f"{n:<20} | {t:<18.3f} | {s:.2f}")
