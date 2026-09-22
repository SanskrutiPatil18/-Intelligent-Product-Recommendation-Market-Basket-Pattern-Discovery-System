
import pandas as pd
from mlxtend.frequent_patterns import association_rules

def generate_association_rules(frequent_itemsets: pd.DataFrame, metric: str = "confidence", min_threshold: float = 0.5, min_lift: float = 1.0):
    """Generates association rules from frequent itemsets and filters them."""
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    filtered_rules = rules[rules['lift'] >= min_lift]
    return filtered_rules.sort_values(by='lift', ascending=False)
