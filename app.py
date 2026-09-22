!pip install fastapi uvicorn pyngrok
from pyngrok import ngrok
# Open a tunnel on port 8000
public_url = ngrok.connect(8000)
print("Public URL:", public_url)


from fastapi import FastAPI, HTTPException
from typing import List
import uvicorn

from .recommender import ProductRecommender
from .schemas import (
    RecommendationRequest, 
    RecommendationResponse, 
    RecommendationItem, 
    RulesResponse, 
    RuleItem, 
    ProductsResponse, 
    AnalyticsResponse
)

app = FastAPI(
    title="Product Recommendation API",
    description="API for Intelligent Product Recommendation and Market Basket Pattern Discovery",
    version="1.0.0"
)

# Initialize the recommender system (load models once at startup)
try:
    recommender = ProductRecommender()
except Exception as e:
    print(f"Failed to initialize recommender: {e}")
    recommender = None # Handle case where models fail to load

@app.get("/health", summary="Health Check", response_model=dict)
async def health_check():
    """Checks if the API is running and recommender models are loaded."""
    if recommender and recommender.association_rules is not None:
        return {"status": "healthy", "models_loaded": True}
    else:
        return {"status": "unhealthy", "models_loaded": False, "detail": "Recommender models not loaded"}

@app.post("/recommend", response_model=RecommendationResponse, summary="Single Product Recommendation")
async def recommend_single_product(request: RecommendationRequest):
    """Generates product recommendations based on a single input product."""
    if not recommender or not recommender.association_rules.empty:
        product_name = request.products[0] if request.products else None
        if not product_name:
            raise HTTPException(status_code=400, detail="Please provide at least one product for recommendation.")
        
        recommendations = recommender.get_single_product_recommendations(product_name)
        return RecommendationResponse(recommendations=recommendations)
    else:
        raise HTTPException(status_code=500, detail="Recommender not initialized or rules are empty.")

@app.post("/recommend-multiple", response_model=RecommendationResponse, summary="Multi-Product Recommendation")
async def recommend_multi_products(request: RecommendationRequest):
    """Generates product recommendations based on multiple input products."""
    if not recommender or not recommender.association_rules.empty:
        if not request.products:
            raise HTTPException(status_code=400, detail="Please provide products for multi-product recommendation.")
        
        recommendations = recommender.get_multi_product_recommendations(request.products)
        return RecommendationResponse(recommendations=recommendations)
    else:
        raise HTTPException(status_code=500, detail="Recommender not initialized or rules are empty.")

@app.get("/rules", response_model=RulesResponse, summary="Retrieve Association Rules")
async def get_rules(
    min_support: float = 0.01, 
    min_confidence: float = 0.5, 
    min_lift: float = 1.0
):
    """Retrieves filtered association rules based on provided thresholds."""
    if not recommender or not recommender.association_rules.empty:
        rules_df = recommender.get_filtered_rules(min_support, min_confidence, min_lift)
        rules_list = []
        for _, row in rules_df.iterrows():
            rules_list.append(RuleItem(
                antecedents=list(row['antecedents']),
                consequents=list(row['consequents']),
                support=row['support'],
                confidence=row['confidence'],
                lift=row['lift'],
                leverage=row['leverage'],
                conviction=row['conviction']
            ))
        return RulesResponse(rules=rules_list)
    else:
        raise HTTPException(status_code=500, detail="Recommender not initialized or rules are empty.")

@app.get("/products", response_model=ProductsResponse, summary="Retrieve Available Products")
async def get_products():
    """Returns a list of all unique products available in the dataset."""
    if recommender:
        products = recommender.get_available_products()
        return ProductsResponse(products=products)
    else:
        raise HTTPException(status_code=500, detail="Recommender not initialized.")

@app.get("/analytics", response_model=AnalyticsResponse, summary="Retrieve Analytics Summary")
async def get_analytics():
    """Returns a summary of key analytics from the dataset and rules."""
    if recommender:
        summary = recommender.get_analytics_summary()
        return AnalyticsResponse(**summary)
    else:
        raise HTTPException(status_code=500, detail="Recommender not initialized.")

