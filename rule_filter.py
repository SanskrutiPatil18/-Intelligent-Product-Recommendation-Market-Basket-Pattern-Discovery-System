
import pandas as pd

def filter_rules(rules_df: pd.DataFrame, min_support: float, min_confidence: float, min_lift: float) -> pd.DataFrame:
    """Filters association rules based on support, confidence, and lift thresholds."""
    filtered_rules_df = rules_df[
        (rules_df['support'] >= min_support) &
        (rules_df['confidence'] >= min_confidence) &
        (rules_df['lift'] >= min_lift)
    ]
    return filtered_rules_df.sort_values(by='lift', ascending=False)
