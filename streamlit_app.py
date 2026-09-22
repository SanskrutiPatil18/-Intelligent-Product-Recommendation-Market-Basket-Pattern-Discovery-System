
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- Configuration ---
FASTAPI_BASE_URL = "http://127.0.0.1:8000" # Change this if your FastAPI server is running elsewhere

# --- Helper Functions to interact with FastAPI ---
def get_products():
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/products")
        response.raise_for_status()
        return response.json()["products"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching products: {e}")
        return []

def get_single_recommendation(product_name):
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/recommend", json={"products": [product_name]})
        response.raise_for_status()
        return response.json()["recommendations"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching single product recommendations: {e}")
        return []

def get_multi_recommendation(product_names):
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/recommend-multiple", json={"products": product_names})
        response.raise_for_status()
        return response.json()["recommendations"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching multi-product recommendations: {e}")
        return []

def get_filtered_rules(min_support, min_confidence, min_lift):
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/rules", params={
            "min_support": min_support,
            "min_confidence": min_confidence,
            "min_lift": min_lift
        })
        response.raise_for_status()
        return response.json()["rules"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching rules: {e}")
        return []

def get_analytics_summary():
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/analytics")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching analytics summary: {e}")
        return None

# --- Streamlit Pages ---
def page_dashboard():
    st.title("📈 Dashboard")
    st.write("Overview of the e-commerce data and association rules.")

    summary = get_analytics_summary()
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", summary.get("total_transactions"))
        with col2:
            st.metric("Total Products", summary.get("total_products"))
        with col3:
            st.metric("Frequent Itemsets", summary.get("num_frequent_itemsets"))
        with col4:
            st.metric("Association Rules", summary.get("num_association_rules"))
        
        st.subheader("Top Selling Combinations")
        for combo in summary.get("top_selling_combinations", []):
            st.write(f"- {combo}")

def page_recommendation():
    st.title("💡 Product Recommendation")
    st.write("Get product recommendations based on selected items.")

    all_products = get_products()

    st.subheader("Single Product Recommendation")
    selected_product_single = st.selectbox("Select a product", [''] + all_products, key='single_product_select')
    if st.button("Get Single Recommendations", key='get_single_reco_btn') and selected_product_single:
        recommendations = get_single_recommendation(selected_product_single)
        if recommendations:
            st.write(f"Recommendations for **{selected_product_single}**:")
            for reco in recommendations:
                st.info(f"**{reco['product']}** (Confidence: {reco['confidence']:.2f}, Lift: {reco['lift']:.2f})")
        else:
            st.write("No recommendations found for this product.")

    st.subheader("Multi-Product Recommendation")
    selected_products_multi = st.multiselect("Select multiple products", all_products, key='multi_product_select')
    if st.button("Get Multi-Recommendations", key='get_multi_reco_btn') and selected_products_multi:
        recommendations = get_multi_recommendation(selected_products_multi)
        if recommendations:
            st.write(f"Recommendations for **{', '.join(selected_products_multi)}**:")
            for reco in recommendations:
                st.info(f"**{reco['product']}** (Confidence: {reco['confidence']:.2f}, Lift: {reco['lift']:.2f})")
        else:
            st.write("No recommendations found for these products.")

def page_rule_explorer():
    st.title("🔎 Association Rule Explorer")
    st.write("Explore and filter association rules based on various metrics.")

    st.sidebar.header("Rule Filtering")
    min_support = st.sidebar.slider("Minimum Support", 0.01, 1.0, 0.02, 0.01)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.01, 1.0, 0.40, 0.01)
    min_lift = st.sidebar.slider("Minimum Lift", 0.0, 10.0, 1.20, 0.1)

    rules = get_filtered_rules(min_support, min_confidence, min_lift)
    if rules:
        rules_df = pd.DataFrame(rules)
        # Convert frozensets back to strings for display if needed
        rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
        
        st.write(f"Found {len(rules_df)} rules with current filters:")
        st.dataframe(rules_df)
    else:
        st.info("No rules found with the specified filters. Try adjusting the thresholds.")

def page_analytics():
    st.title("📊 Analytics & Visualizations")
    st.write("Detailed visualizations of product relationships.")
    st.info("This section would typically include various plots and charts using libraries like Plotly or Matplotlib, visualizing product frequency, top combinations, support vs confidence, etc. For this example, we'll keep it simple.")

    # Example: Plotting a dummy chart or showing some aggregated data
    summary = get_analytics_summary()
    if summary:
        # Dummy data for demonstration
        data = {'Metric': ['Total Transactions', 'Total Products', 'Frequent Itemsets', 'Association Rules'],
                'Value': [summary.get('total_transactions', 0),
                          summary.get('total_products', 0),
                          summary.get('num_frequent_itemsets', 0),
                          summary.get('num_association_rules', 0)]}
        df_metrics = pd.DataFrame(data)
        
        fig = px.bar(df_metrics, x='Metric', y='Value', title='Key Metrics Overview')
        st.plotly_chart(fig)

def page_algorithm_comparison():
    st.title("⚙️ Algorithm Comparison")
    st.write("Compare the performance of Apriori and FP-Growth.")
    st.warning("To implement this page fully, you would need to expose the comparison data (runtime, itemsets count) from the FastAPI backend. For now, this is a placeholder.")
    st.info("This page would display metrics like runtime, number of frequent itemsets, and memory usage for both Apriori and FP-Growth, ideally fetched from a backend endpoint.")

# --- Main App Logic ---
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", [
    "Dashboard",
    "Product Recommendation",
    "Association Rule Explorer",
    "Analytics",
    "Algorithm Comparison"
])

if selection == "Dashboard":
    page_dashboard()
elif selection == "Product Recommendation":
    page_recommendation()
elif selection == "Association Rule Explorer":
    page_rule_explorer()
elif selection == "Analytics":
    page_analytics()
elif selection == "Algorithm Comparison":
    page_algorithm_comparison()
