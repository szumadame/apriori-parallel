import pandas as pd
from collections import defaultdict
from itertools import combinations


data = {
    "Transaction ID": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"],
    "Categories": [
        "Fiction, Mystery, Romance",
        "Sci-Fi, History",
        "Fiction, Sci-Fi, Biography",
        "History, Biography",
        "Mystery, Sci-Fi, History",
        "Fiction, Sci-Fi",
        "Fiction, Mystery, History, Sci-Fi",
        "Romance, Biography",
        "Fiction, Mystery, Sci-Fi",
        "Sci-Fi, History"
    ]
}

df = pd.DataFrame(data)

min_support_threshold = 0.2
min_confidence_threshold = 0.7


def calculate_support(df, itemset_col, candidate_itemsets):
    """
    Calculate support for given n-itemsets.

    Parameters:
        df (pd.DataFrame): The DataFrame containing transactions.
        itemset_col (str): The column name where itemsets (as comma-separated strings) are stored.
        candidate_itemsets (list of tuples): List of itemsets (tuples of items) for which to compute support.

    Returns:
        dict: A dictionary mapping each itemset to its support value.
    """
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
    """
    Optimized candidate generation using frequent (k-1)-itemsets.

    Parameters:
        pruned_support (dict): Frequent (k-1)-itemsets with their supports.
        k (int): Size of itemsets to generate (k-itemsets).

    Returns:
        list of tuples: Candidate k-itemsets.
    """
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
    """
    Prunes candidates from itemset that don't meet support treshold.

    Parameters:
        support (dict): The pruned itemsets with their support.
        minimal_support_treshold (int): Minimal support treshold itemset needs to have not to be pruned.

    Returns:
        dict: A dictionary mapping each itemset to its support value.
    """
    return {k: v for k, v in support.items() if v >= minimal_support_treshold}

def generate_association_rules(all_supports, min_confidence_threshold):
    """
    Generates association rules from all itemsets and calculates confidence.

    Parameters:
        all_supports (dict): A dictionary with itemsets of all sizes and their supports.
        min_confidence_threshold (float): Minimum confidence threshold.

    Returns:
        list of tuples: (antecedent, consequent, confidence)
    """
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

if __name__ == "__main__":
  total_transactions = len(df)
  all_items = set()
  for categories in df["Categories"]:
    all_items.update(categories.split(", "))
  
  itemset_col = 'Categories'
  min_support_threshold = 0.4
  min_confidence_threshold = 0.6

  all_supports = {}
  size = 1

  # Step 1: Initial 1-itemsets
  pruned_support = prune(
    calculate_support(df, itemset_col, [(item,) for item in all_items]),
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

    support_itemsets = calculate_support(df, itemset_col, candidate_itemsets)
    pruned_support = prune(support_itemsets, min_support_threshold)

    if pruned_support:
        print(f"Pruned support after {size}-itemsets:")
        print(pruned_support)
        all_supports.update(pruned_support)
    else:
        break
    
  rules = generate_association_rules(all_supports, min_confidence_threshold)
  print("\nAssociation Rules:")
  for antecedent, consequent, confidence in rules:
      print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence:.2f}")