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

def generate_candidates(pruned_support, n):
    """
    Generate candidate n-itemsets based on pruned (n-1)-itemsets.

    Parameters:
        pruned_support (dict): The pruned itemsets with their support.
        n (int): The size of itemsets to generate.

    Returns:
        list of tuples: A list of candidate n-itemsets.
    """
    # Extract itemsets from pruned support keys
    items = set()
    for itemset, support in pruned_support.items():
        items.update(itemset)

    if n == 1:
        # Return the original pruned 1-itemsets
        return [(item,) for item in items]

    # For n > 1, generate n-item combinations from the flattened individual items
    candidate_n_itemsets = [tuple(sorted(comb)) for comb in combinations(items, n)]
    
    return candidate_n_itemsets

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

  candidate_1_itemsets = [(item,) for item in all_items]
  support = calculate_support(df, itemset_col, candidate_1_itemsets)
  pruned_support = prune(support, min_support_threshold)

  print('Pruned support po 1 itemsecie')
  print(pruned_support)

  #Tworzymy podwojnego kandydata
  candidate_2_itemsets  = generate_candidates(pruned_support, 2)
  support_2_itemsets = calculate_support(df, itemset_col, candidate_2_itemsets)
  pruned_support_2 = prune(support_2_itemsets, min_support_threshold)
  print('Pruned support po 2 itemsecie')
  print(pruned_support_2)

  #Tworzymy potrojnego kandydata
  candidate_3_itemsets  = generate_candidates(pruned_support_2, 3)
  support_3_itemsets = calculate_support(df, itemset_col, candidate_3_itemsets)

  pruned_support_3 = prune(support_3_itemsets, min_support_threshold)
  print(support_3_itemsets)
  print('Pruned support po 3 itemsecie')
  print(pruned_support_3)

  all_supports = {}
  all_supports.update(pruned_support)
  all_supports.update(pruned_support_2)
  all_supports.update(pruned_support_3)

  rules = generate_association_rules(all_supports, 0.7)

  for antecedent, consequent, confidence in rules:
    print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence:.2f}")
