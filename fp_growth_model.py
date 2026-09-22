
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

def run_fpgrowth(transactions_onehot_df: pd.DataFrame, min_support: float = 0.01):
    """Applies the FP-Growth algorithm to find frequent itemsets."""
    frequent_itemsets = fpgrowth(transactions_onehot_df, min_support=min_support, use_colnames=True)
    return frequent_itemsets.sort_values(by='support', ascending=False)
