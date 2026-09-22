
import pandas as pd
import time
from mlxtend.frequent_patterns import apriori, fpgrowth

def evaluate_algorithms(transactions_onehot_df: pd.DataFrame, min_support: float = 0.01):
    """Compares the runtime and number of frequent itemsets for Apriori and FP-Growth."""
    start_time = time.time()
    frequent_itemsets_apriori = apriori(transactions_onehot_df, min_support=min_support, use_colnames=True)
    apriori_runtime = time.time() - start_time

    start_time = time.time()
    frequent_itemsets_fpgrowth = fpgrowth(transactions_onehot_df, min_support=min_support, use_colnames=True)
    fpgrowth_runtime = time.time() - start_time

    comparison_data = {
        'Algorithm': ['Apriori', 'FP-Growth'],
        'Runtime (s)': [apriori_runtime, fpgrowth_runtime],
        'Number of Frequent Itemsets': [len(frequent_itemsets_apriori), len(frequent_itemsets_fpgrowth)]
    }
    return pd.DataFrame(comparison_data)
