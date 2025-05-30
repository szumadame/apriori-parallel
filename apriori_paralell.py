import os
import pprint
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool, cpu_count


def calculate_support(df, itemset_col, candidate_itemsets):
    total_transactions = len(df)
    support_counts = {}

    for itemset in candidate_itemsets:
        count = 0
        for transaction in df[itemset_col]:
            items = set(transaction.split(", "))
            if set(itemset).issubset(items):
                count += 1
        support_counts[itemset] = count / total_transactions

    return support_counts

def generate_candidates(pruned_support, k):
    prev_frequent_itemsets = list(pruned_support.keys())
    candidates = set()

    for i in range(len(prev_frequent_itemsets)):
        for j in range(i + 1, len(prev_frequent_itemsets)):
            l1 = prev_frequent_itemsets[i]
            l2 = prev_frequent_itemsets[j]

            # Join step: combine if first (k-2) items are equal
            if l1[:k-2] == l2[:k-2]:
                candidate = tuple(sorted(set(l1) | set(l2)))
                if len(candidate) == k:
                    # Prune step: all (k-1)-subsets must be frequent
                    subsets = combinations(candidate, k - 1)
                    if all(tuple(sorted(sub)) in pruned_support for sub in subsets):
                        candidates.add(candidate)

    return list(candidates)

def prune(support, minimal_support_treshold):
    return {k: v for k, v in support.items() if v >= minimal_support_treshold}

def generate_association_rules(all_supports, min_confidence_threshold):
    rules = []
    for itemset in all_supports:
        if len(itemset) < 2:
            continue  # Cannot generate rules from 1-itemsets

        itemset_support = all_supports[itemset]

        # Try all possible non-empty antecedents
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if not consequent:
                    continue  # skip empty consequents

                if antecedent in all_supports:
                    antecedent_support = all_supports[antecedent]
                    if antecedent_support > 0:
                      #Counting confidence = eg. support(Fiction, Romance) / support(Fiction) 
                        confidence = itemset_support / antecedent_support
                        if confidence >= min_confidence_threshold:
                            rules.append((antecedent, consequent, confidence))
    
    return rules


def calculate_partial_support(args):
    partition_idx, partition_df, itemset_col, candidate_itemsets = args

    pid = os.getpid()  # Process ID
    print(f"[Process {pid}] Starting partition {partition_idx} with {len(partition_df)} transactions.")  
    #print(partition_df)
    return calculate_support(partition_df, itemset_col, candidate_itemsets)


def calculate_support_parallel(df, itemset_col, candidate_itemsets, n_partitions=None):
    print("Partitions used: ", n_partitions)
    if n_partitions is None:
        n_partitions = cpu_count()

    partitions = np.array_split(df, n_partitions)
    args = [(idx, partition, itemset_col, candidate_itemsets) for idx, partition in enumerate(partitions)]

    with Pool(processes=n_partitions) as pool:
        local_supports_list = pool.map(calculate_partial_support, args)

    # aggregate partial counts (weighted by partition sizes)
    total_transactions = len(df)
    aggregated_counts = {}
    for local_supports, partition in zip(local_supports_list, partitions):
        partition_size = len(partition)
        for itemset, local_support in local_supports.items():
            # convert local support to counts
            local_count = local_support * partition_size
            aggregated_counts[itemset] = aggregated_counts.get(itemset, 0) + local_count

    # convert back to global support
    final_support = {itemset: count / total_transactions for itemset, count in aggregated_counts.items()}
    return final_support


if __name__ == "__main__":
  df = pd.read_pickle("/workspaces/apriori-parallel/data/grocery_df.pkl")
  total_transactions = len(df)

  itemset_col = 'Categories'
  all_items = set()
  for categories in df[itemset_col]:
    all_items.update(categories.split(", "))
  
  min_support_threshold = 0.3
  min_confidence_threshold = 0.5

  all_supports = {}
  size = 1

  n_partitions=25

  # Step 1: Initial 1-itemsets
  pruned_support = prune(
    calculate_support_parallel(df, itemset_col, [(item,) for item in all_items], n_partitions),
    min_support_threshold
  )
  all_supports.update(pruned_support)
  print(f"Pruned support after 1-itemsets:")
  print(pruned_support)

  # Step 2: Iteratively build larger itemsets
  while pruned_support:
    size += 1
    candidate_itemsets = generate_candidates(pruned_support, size)
    if not candidate_itemsets:
        break

    support_itemsets = calculate_support_parallel(df, itemset_col, candidate_itemsets, n_partitions)
    pruned_support = prune(support_itemsets, min_support_threshold)

    if pruned_support:
        print(f"Pruned support after {size}-itemsets:")
        print(pruned_support)
        all_supports.update(pruned_support)
    else:
        break

  sorted_supports = dict(
    sorted(
        all_supports.items(),
        key=lambda x: (len(x[0]), x[0])
    )
  )

  pp = pprint.PrettyPrinter(indent=2, width=100)
  print("========= All Supports =========")
  pp.pprint(sorted_supports)

  rules = generate_association_rules(all_supports, min_confidence_threshold)
  print("\nAssociation Rules:")
  for antecedent, consequent, confidence in rules:
      print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence:.2f}")