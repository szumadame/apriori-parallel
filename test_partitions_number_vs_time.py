import time
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint

from apriori_parallel import run_apriori

if __name__ == "__main__":
    df_name = "mushroom_apriori"
    df = pd.read_pickle(f"/workspaces/apriori-parallel/data/{df_name}_df.pkl")
    itemset_col = "Categories"

    min_support_threshold = 0.6
    min_confidence_threshold = 0.7

    partition_counts = [1, 2, 4, 8, 16, 32, 64]
    n_iterations = 5

    #execution_times = []
    execution_times_dict = {p: [] for p in partition_counts}

    for iteration in range(1, n_iterations + 1):
        print(f"\n====== Iteration {iteration}/{n_iterations} ======")
        for n_partitions in partition_counts:
            print(f"\nRunning Apriori with {n_partitions} partitions...")
            start_time = time.time()

            all_supports, rules = run_apriori(
                df,
                itemset_col,
                min_support_threshold,
                min_confidence_threshold,
                n_partitions
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Execution time with {n_partitions} partitions: {elapsed_time:.2f} seconds")
            execution_times_dict[n_partitions].append(elapsed_time)
        # print(f"Execution time with {n_partitions} partitions: {elapsed_time:.2f} seconds")
        # execution_times.append(elapsed_time)
    
    avg_execution_times = {
        p: np.mean(execution_times_dict[p]) for p in partition_counts
    }

    results_df = pd.DataFrame({
        "Partitions": partition_counts,
        "Average Execution Time (s)": [avg_execution_times[p] for p in partition_counts]
    })

    plt.figure(figsize=(10, 6))
    plt.plot(results_df["Partitions"], results_df["Average Execution Time (s)"],
             marker='o', color='b', label='Average Execution Time')
    plt.xlabel("Number of Partitions")
    plt.ylabel("Average Execution Time (seconds)")
    plt.title(f"Average Execution Time over {n_iterations} Runs\nApriori on {df_name} Dataset")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/execution_time/avg_execution_time_{df_name}_supp={min_support_threshold}conf={min_confidence_threshold}.png")
    plt.show()

    execution_times_df = pd.DataFrame(execution_times_dict)
    execution_times_df.index = [f"Run {i+1}" for i in range(n_iterations)]

    # results_df.to_csv(
    #     f"results/avg_execution_times_grocery_supp={min_support_threshold}conf={min_confidence_threshold}.csv",
    #     index=False
    # )

    pp = pprint.PrettyPrinter(indent=2, width=100)
    print("\nDetailed Execution Times (seconds):")
    print(execution_times_df)
    print("\nAverage Execution Times:")
    print(results_df)

    print("\nFinal Association Rules (from last run!):")
    for antecedent, consequent, confidence in rules:
        print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence:.2f}")

    data = []
    for antecedent, consequent, confidence in rules:
        data.append({
            'Antecedent': ', '.join(antecedent),
            'Consequent': ', '.join(consequent),
            'Confidence': confidence
        })

    rules_df = pd.DataFrame(data)
    rules_df.to_csv(
        f"results/execution_time/final_association_rules_{df_name}_supp={min_support_threshold}conf={min_confidence_threshold}.csv",
        index=False
    )