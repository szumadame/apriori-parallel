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


def calculate_support_count_and_support(data, total, size=1):
    result = []
    for combo in combinations(set().union(*data), size):  # Ensure we're working with unique items
        combo_set = set(combo)  # Convert combination to a set for easy checking
        support_count = sum(1 for transaction in data if combo_set.issubset(transaction))
        support = support_count / total
        result.append({
            "product": combo,  # Store the combination as the "product"
            "support": support,
            "support_count": support_count
        })
    return result


def filter_by_min_support(support_data, min_support):
    return [entry for entry in support_data if entry["support"] > min_support]



if __name__=="__main__":
    total_transactions = len(dataset)
    transactions = [set(items) for items in dataset.values()]

    size = 1
    frequent_itemsets = []
    current_itemsets = None

    supports = calculate_support_count_and_support(transactions, total_transactions, 2)
    check_supports = filter_by_min_support(supports, min_support_treshold)
    
    
    print(supports)
    print(check_supports)