import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


file_path = "/workspaces/apriori-parallel/data/grocery.csv"

#Creating dataframe from csv


# with open(file_path, "r") as file:
#     lines = file.readlines()

# lines = [line.strip() for line in lines if line.strip()]

# df = pd.DataFrame({
#     "Transaction ID": [f"T{i+1}" for i in range(len(lines))],
#     "Categories": [", ".join(item.strip() for item in line.split(",")) for line in lines]
# })

#df.to_pickle("/workspaces/apriori-parallel/data/grocery_df.pkl")




#Checking frequent rules for the dataset using Apriori library

df = pd.read_pickle("/workspaces/apriori-parallel/data/grocery_df.pkl")

dataset = df['Categories'].apply(lambda x: [item.strip() for item in x.split(",")]).tolist()
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df_encoded  = pd.DataFrame(te_ary, columns=te.columns_)

#print(df_encoded[:5])

minsup = 0.25 
frequent_itemsets = apriori(df_encoded, min_support=minsup, use_colnames=True)

if frequent_itemsets.empty:
    print("No frequent itemsets found at this support threshold.")
else:
    print("Frequent Itemsets:")
    print(frequent_itemsets)

    minconf = 0.5
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    print("\nAssociation Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
