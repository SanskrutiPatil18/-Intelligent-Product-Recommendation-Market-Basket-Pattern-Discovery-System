
import pandas as pd

def get_analytics_summary_util(frequent_itemsets: pd.DataFrame, association_rules: pd.DataFrame, product_encoder_columns: list, top_n_combinations: int = 5) -> dict:
    """Provides a summary of key analytics."""
    total_products = len(product_encoder_columns)
    num_frequent_itemsets = len(frequent_itemsets)
    num_association_rules = len(association_rules)

    # Assuming total transactions can be derived from the product_encoder if it was trained on unique transactions.
    # This part needs actual transaction_id count if available from raw data.
    # For now, let's derive it from the range of transaction_ids in the original dataframe.
    # This function expects product_encoder_columns and frequent_itemsets/association_rules.

    # Top-selling combinations (from frequent itemsets)
    top_selling_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).head(top_n_combinations)
    top_combinations = [str(list(itemset)) for itemset in top_selling_itemsets['itemsets']]

    return {
        "total_products": total_products,
        "num_frequent_itemsets": num_frequent_itemsets,
        "num_association_rules": num_association_rules,
        "top_selling_combinations": top_combinations
    }
