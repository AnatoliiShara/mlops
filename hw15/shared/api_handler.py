# shared/api_handler.py
# Простий API handler для мульти-модель інференсу

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import os
from book_models import create_multi_model_predictor

# Pydantic моделі для API
class PredictionRequest(BaseModel):
    model_name: str
    input_data: Dict[str, Any]

class PredictionResponse(BaseModel):
    result: Dict[str, Any]
    model_info: Dict[str, str]

class HealthResponse(BaseModel):
    status: str
    models: Dict[str, Dict[str, str]]

# Створення FastAPI додатку
app = FastAPI(
    title="Multi-Model Book Recommendation API",
    description="Simple multi-model deployment for book recommendations",
    version="1.0.0"
)

# Глобальний предиктор
predictor = None

@app.on_event("startup")
async def startup_event():
    """Ініціалізація моделей при старті"""
    global predictor
    predictor = create_multi_model_predictor()
    print("✅ Multi-model predictor initialized")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Model Book Recommendation API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    return predictor.health_check()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Multi-model prediction endpoint"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        result = predictor.predict(request.model_name, request.input_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return PredictionResponse(
            result=result,
            model_info={
                "requested_model": request.model_name,
                "api_version": "1.0.0"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=Dict[str, List[str]])
async def list_models():
    """Список доступних моделей"""
    return {
        "available_models": [
            "embedding", 
            "recommendation", 
            "quality", 
            "ensemble"
        ],
        "description": {
            "embedding": "Text embedding generation",
            "recommendation": "Book recommendations",
            "quality": "Recommendation quality scoring",
            "ensemble": "Combined pipeline"
        }
    }

# Специфічні endpoints для кожної моделі (для зручності)
@app.post("/embedding")
async def embedding_endpoint(texts: List[str]):
    """Спеціальний endpoint для embeddings"""
    request = PredictionRequest(
        model_name="embedding",
        input_data={"texts": texts}
    )
    return await predict(request)

@app.post("/recommendation")  
async def recommendation_endpoint(query: str, top_k: int = 3):
    """Спеціальний endpoint для рекомендацій"""
    request = PredictionRequest(
        model_name="recommendation",
        input_data={"query": query, "top_k": top_k}
    )
    return await predict(request)

@app.post("/quality")
async def quality_endpoint(query: str, recommendations: List[Dict]):
    """Спеціальний endpoint для оцінки якості"""
    request = PredictionRequest(
        model_name="quality", 
        input_data={"query": query, "recommendations": recommendations}
    )
    return await predict(request)

@app.post("/ensemble")
async def ensemble_endpoint(query: str, top_k: int = 3):
    """Спеціальний endpoint для ensemble"""
    request = PredictionRequest(
        model_name="ensemble",
        input_data={"query": query, "top_k": top_k}
    )
    return await predict(request)

# Для локального запуску
def run_local():
    """Запуск API локально"""
    uvicorn.run(
        "api_handler:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    run_local()