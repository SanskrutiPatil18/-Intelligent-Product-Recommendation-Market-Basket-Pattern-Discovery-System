
from typing import List, Optional
from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    products: List[str]

class RecommendationItem(BaseModel):
    product: str
    confidence: float
    lift: float

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]

class RuleItem(BaseModel):
    antecedents: List[str]
    consequents: List[str]
    support: float
    confidence: float
    lift: float
    leverage: float
    conviction: float

class RulesResponse(BaseModel):
    rules: List[RuleItem]

class ProductsResponse(BaseModel):
    products: List[str]

class AnalyticsResponse(BaseModel):
    total_transactions: int
    total_products: int
    num_frequent_itemsets: int
    num_association_rules: int
    top_selling_combinations: List[str]
