from itertools import combinations


dataset = {
    "1": ["Bread", "Butter", "Milk"],
    "2": ["Bread", "Butter"],
    "3": ["Bread", "Milk"],
    "4": ["Butter", "Milk"],
    "5": ["Bread", "Milk"]
}

min_support_treshold = 0.5
min_confidence_treshold = 0.7


def calculate_support_count_and_support(data, total, candidates):
    result = []
    for combo in candidates:
        combo_set = set(combo)
        support_count = sum(1 for transaction in data if combo_set.issubset(transaction))
        support = support_count / total
        result.append({
            "product": combo,
            "support": support,
            "support_count": support_count
        })
    return result

def filter_by_min_support(support_data, min_support):
    return [entry for entry in support_data if entry["support"] > min_support]

def generate_candidates(prev_frequent_itemsets, k):
    """Join step of Apriori: combine (k-1)-itemsets to make k-itemsets"""
    items = [set(item['product']) for item in prev_frequent_itemsets]
    candidates = set()
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            union = items[i] | items[j]
            if len(union) == k:
                candidates.add(tuple(sorted(union)))
    return list(candidates)


if __name__=="__main__":
    total_transactions = len(dataset)
    transactions = [set(items) for items in dataset.values()]
    frequent_itemsets = []
    size = 1
    
    single_items = sorted(set().union(*transactions))
    candidates = [(item,) for item in single_items]


    while candidates:
        # Step 2: Calculate support for candidates
        support_data = calculate_support_count_and_support(transactions, total_transactions, candidates)

        # Step 3: Filter by min support
        current_frequents = filter_by_min_support(support_data, min_support_treshold)
        if not current_frequents:
            break

        frequent_itemsets.extend(current_frequents)

        # Step 4: Generate next size candidates from current frequents
        size += 1
        candidates = generate_candidates(current_frequents, size)

    # Print result
for itemset in frequent_itemsets:
    print(itemset)