from ucimlrepo import fetch_ucirepo 
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#Creating dataframe from library dataset


# mushroom = fetch_ucirepo(id=73) 
  
# X = mushroom.data.features 
# y = mushroom.data.targets 
  
# transactions = []
# for idx, row in X.iterrows():
#     transaction = [f"{col}={row[col]}" for col in X.columns]
#     transactions.append(transaction)

# df_apriori = pd.DataFrame({
#     "Transaction ID": [f"T{i+1}" for i in range(len(transactions))],
#     "Categories": [", ".join(t) for t in transactions]
# })


# df_apriori.to_pickle("/workspaces/apriori-parallel/data/mushroom_apriori_df.pkl")
# print(df_apriori.head())




#Checking frequent rules for the dataset using Apriori library

df = pd.read_pickle("/workspaces/apriori-parallel/data/mushroom_apriori_df.pkl")

dataset = df['Categories'].apply(lambda x: [item.strip() for item in x.split(",")]).tolist()
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df_encoded  = pd.DataFrame(te_ary, columns=te.columns_)

minsup = 0.9 
frequent_itemsets = apriori(df_encoded, min_support=minsup, use_colnames=True)

if frequent_itemsets.empty:
    print("No frequent itemsets found at this support threshold.")
else:
    print("Frequent Itemsets:")
    print(frequent_itemsets)

    minconf = 0.6
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    print("\nAssociation Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
