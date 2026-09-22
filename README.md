
# Intelligent Product Recommendation & Market Basket Pattern Discovery System

This repository contains a full-stack solution for product recommendation and market basket analysis, utilizing Association Rule Mining. The system includes data processing, model training (Apriori and FP-Growth), a FastAPI backend for serving recommendations, and a Streamlit frontend for interactive exploration.

## Project Structure

```
market-basket-recommendation/
├── data/
│   └── transactions.csv
├── models/
│   ├── association_rules.pkl
│   ├── frequent_itemsets.pkl
│   ├── product_encoder.pkl
│   └── product_mapping.pkl
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── recommender.py
│   └── schemas.py
├── frontend/
│   └── streamlit_app.py
├── outputs/
│   └── generated_rules.csv
├── training/
│   ├── apriori_model.py
│   ├── evaluate.py
│   ├── fp_growth_model.py
│   ├── preprocess.py
│   └── rule_generation.py
├── utils/
│   ├── analytics.py
│   ├── __init__.py
│   ├── rule_filter.py
│   └── visualization.py
├── requirements.txt
├── README.md
├── .gitignore
└── Intelligent Product Recommendation & Market Basket Pattern Discovery System.ipynb (This Colab Notebook)
```

## Setup and Installation

1.  **Clone the repository (or download the files generated in this Colab notebook).**

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data:** Ensure `transactions.csv` is placed in the `data/` directory.

## How to Run

To run the full system, you need to start both the FastAPI backend and the Streamlit frontend concurrently. You can do this by opening two separate terminal windows or processes.

### 1. Start the FastAPI Backend

Navigate to the `backend` directory and run the `uvicorn` server:

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

-   `app:app`: Refers to the `app` object inside `app.py`.
-   `--host 0.0.0.0`: Makes the server accessible from outside the local machine (useful in environments like Colab or Docker).
-   `--port 8000`: Specifies the port number.
-   `--reload`: Automatically reloads the server on code changes (useful for development).

You should see output indicating that the FastAPI application is running.

### 2. Start the Streamlit Frontend

In a separate terminal, navigate to the `frontend` directory and run the Streamlit application:

```bash
cd frontend
streamlit run streamlit_app.py
```

Streamlit will typically open a new tab in your web browser with the application. If not, it will provide a local URL (e.g., `http://localhost:8501`) and a network URL.

## Features

*   **Product Recommendation**: Get recommendations based on single or multiple selected products.
*   **Association Rule Explorer**: Browse and filter discovered association rules.
*   **Dashboard & Analytics**: View key metrics, top-selling combinations, and visualizations.
*   **Algorithm Comparison**: See performance metrics for Apriori and FP-Growth algorithms.

## Contributing

Feel free to fork the repository, open issues, or submit pull requests.
