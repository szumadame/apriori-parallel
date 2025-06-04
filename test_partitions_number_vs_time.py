import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import pprint

from apriori_parallel import calculate_support_parallel, prune, generate_candidates, generate_association_rules

def test_apriori_with_partitions(df, itemset_col, min_support_threshold, min_confidence_threshold):
    all_items = set()
    for categories in df[itemset_col]:
        all_items.update(categories.split(", "))

    all_supports = {}
    size = 1

    # Initial 1-itemsets
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


if __name__ == "__main__":
    df = pd.read_pickle("/workspaces/apriori-parallel/data/chicago_crime_df.pkl")
    itemset_col = "Categories"

    min_support_threshold = 0.01
    min_confidence_threshold = 0.4

    partition_counts = [1, 2, 3]
    execution_times = []

    for n_partitions in partition_counts:
        print(f"\nRunning Apriori with {n_partitions} partitions...")
        start_time = time.time()

        globals()["n_partitions"] = n_partitions
        all_supports, rules = test_apriori_with_partitions(
            df,
            itemset_col,
            min_support_threshold,
            min_confidence_threshold
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time with {n_partitions} partitions: {elapsed_time:.2f} seconds")
        execution_times.append(elapsed_time)

    results_df = pd.DataFrame({
        "Partitions": partition_counts,
        "Execution Time (s)": execution_times
    })

    plt.figure(figsize=(10, 6))
    plt.plot(results_df["Partitions"], results_df["Execution Time (s)"], marker='o', color='b')
    plt.xlabel("Number of Partitions")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Influence of Number of Partitions on Apriori Execution Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/execution_time_plot_chicago_crime_supp={min_support_threshold}conf={min_confidence_threshold}.png")
    plt.show()

    pp = pprint.PrettyPrinter(indent=2, width=100)
    print("\nExecution Times by Partition Count:")
    print(results_df)
    pp.pprint(all_supports)
    pp.pprint(rules)

    data = []
    for antecedent, consequent, confidence in rules:
        data.append({
        'Antecedent': ', '.join(antecedent),
        'Consequent': ', '.join(consequent),
        'Confidence': confidence
        })

    df = pd.DataFrame(data)
    df.to_csv(f"results/association_rules_chicago_crime_supp={min_support_threshold}conf={min_confidence_threshold}.csv", index=False)