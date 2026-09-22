
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

def plot_algorithm_comparison(comparison_df: pd.DataFrame):
    """Plots the runtime and number of frequent itemsets for algorithm comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x='Algorithm', y='Runtime (s)', data=comparison_df, ax=axes[0])
    axes[0].set_title('Algorithm Runtime Comparison')
    sns.barplot(x='Algorithm', y='Number of Frequent Itemsets', data=comparison_df, ax=axes[1])
    axes[1].set_title('Number of Frequent Itemsets Found')
    plt.tight_layout()
    plt.show()

def plot_rule_metrics(rules_df: pd.DataFrame):
    """Plots support, confidence, and lift distributions of association rules."""
    fig = px.scatter(rules_df, x="support", y="confidence",
                     size="lift", color="lift", hover_name="antecedents",
                     title="Association Rules: Support vs Confidence (sized by Lift)")
    fig.show()
