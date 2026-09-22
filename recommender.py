
import joblib
import pandas as pd
import os

class ProductRecommender:
    def __init__(self, models_path='models'):
        self.models_path = models_path
        self._load_models()

    def _load_models(self):
        try:
            self.association_rules = joblib.load(os.path.join(self.models_path, 'association_rules.pkl'))
            self.frequent_itemsets = joblib.load(os.path.join(self.models_path, 'frequent_itemsets.pkl'))
            self.product_encoder = joblib.load(os.path.join(self.models_path, 'product_encoder.pkl'))
            self.product_mapping = joblib.load(os.path.join(self.models_path, 'product_mapping.pkl'))
            print("Models loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading model file: {e}. Make sure models are trained and saved in the '{self.models_path}' directory.")
            raise
        
        # Convert frozensets in 'antecedents' and 'consequents' to sorted lists of strings for easier use
        self.association_rules['antecedents_str'] = self.association_rules['antecedents'].apply(lambda x: sorted(list(x)))
        self.association_rules['consequents_str'] = self.association_rules['consequents'].apply(lambda x: sorted(list(x)))

    def get_available_products(self) -> list:
        return sorted(list(self.product_encoder.columns_))

    def _get_recommendations_for_antecedent(self, current_products: set, top_n: int = 5) -> list:
        # Ensure current_products is a frozenset for matching
        current_products_frozenset = frozenset(current_products)

        # Find rules where current_products are the antecedents
        relevant_rules = self.association_rules[
            self.association_rules['antecedents'].apply(lambda x: current_products_frozenset.issubset(x))
        ].copy()

        if relevant_rules.empty:
            return []

        # Filter out recommendations that are already in current_products
        relevant_rules['consequents_to_recommend'] = relevant_rules['consequents'].apply(
            lambda x: frozenset(item for item in x if item not in current_products_frozenset)
        )
        
        # Drop rows where there are no new consequents to recommend
        relevant_rules = relevant_rules[relevant_rules['consequents_to_recommend'].apply(len) > 0]

        if relevant_rules.empty:
            return []

        # Sort by lift, then confidence, then support
        relevant_rules = relevant_rules.sort_values(by=['lift', 'confidence', 'support'], ascending=False)

        recommendations = []
        recommended_products_set = set()

        for _, row in relevant_rules.iterrows():
            for product in row['consequents_to_recommend']:
                if product not in recommended_products_set:
                    recommendations.append({
                        "product": product,
                        "confidence": row['confidence'],
                        "lift": row['lift']
                    })
                    recommended_products_set.add(product)
                    if len(recommendations) >= top_n:
                        return recommendations
        return recommendations

    def get_single_product_recommendations(self, product_name: str, top_n: int = 5) -> list:
        return self._get_recommendations_for_antecedent(frozenset([product_name]), top_n)

    def get_multi_product_recommendations(self, product_names: List[str], top_n: int = 5) -> list:
        return self._get_recommendations_for_antecedent(frozenset(product_names), top_n)

    def get_filtered_rules(self, min_support: float = 0.01, min_confidence: float = 0.5, min_lift: float = 1.0) -> pd.DataFrame:
        filtered_rules_df = self.association_rules[
            (self.association_rules['support'] >= min_support) &
            (self.association_rules['confidence'] >= min_confidence) &
            (self.association_rules['lift'] >= min_lift)
        ].copy()
        return filtered_rules_df.sort_values(by='lift', ascending=False)
    
    def get_analytics_summary(self) -> dict:
        total_transactions = len(self.product_encoder.inverse_transform(pd.DataFrame([False]*len(self.product_encoder.columns_)).T))
        total_products = len(self.product_encoder.columns_)
        num_frequent_itemsets = len(self.frequent_itemsets)
        num_association_rules = len(self.association_rules)
        
        # Top-selling combinations (from frequent itemsets)
        top_selling_itemsets = self.frequent_itemsets.sort_values(by='support', ascending=False).head(5)
        top_combinations = [str(list(itemset)) for itemset in top_selling_itemsets['itemsets']]
        
        return {
            "total_transactions": total_transactions,
            "total_products": total_products,
            "num_frequent_itemsets": num_frequent_itemsets,
            "num_association_rules": num_association_rules,
            "top_selling_combinations": top_combinations
        }

