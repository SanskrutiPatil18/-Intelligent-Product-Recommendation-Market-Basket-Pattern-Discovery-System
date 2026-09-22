
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

def preprocess_transactions(transactions_df: pd.DataFrame):
    """Groups transactions and performs one-hot encoding."""
    transactions_list = transactions_df.groupby('transaction_id')['product'].apply(list).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions_list).transform(transactions_list)
    transactions_onehot_df = pd.DataFrame(te_ary, columns=te.columns_)
    return transactions_onehot_df, te
