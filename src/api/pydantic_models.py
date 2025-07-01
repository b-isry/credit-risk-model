from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    total_transaction_amount: float
    avg_transaction_amount: float
    std_transaction_amount: float
    transaction_count: int
    unique_products_purchased: int
    unique_providers_used: int
    Recency: int
    Frequency: int
    Monetary: float
    
    class Config:
        schema_extra = {
            "example": {
                "total_transaction_amount": 1500.75,
                "avg_transaction_amount": 75.04,
                "std_transaction_amount": 20.50,
                "transaction_count": 20,
                "unique_products_purchased": 5,
                "unique_providers_used": 3,
                "Recency": 30,
                "Frequency": 15,
                "Monetary": 1200.00
            }
        }
        
class PredictionOutput(BaseModel):
    customer_id: str | None = None
    risk_probability: float = Field(..., ge=0, le=1)
    is_high_risk: bool
    