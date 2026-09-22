
import pandas as pd
from mlxtend.frequent_patterns import apriori

def run_apriori(transactions_onehot_df: pd.DataFrame, min_support: float = 0.01):
    """Applies the Apriori algorithm to find frequent itemsets."""
    frequent_itemsets = apriori(transactions_onehot_df, min_support=min_support, use_colnames=True)
    return frequent_itemsets.sort_values(by='support', ascending=False)
