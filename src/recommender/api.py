"""Small FastAPI demo exposing the recommender for local testing.

Author: rishabhrising
Course: AI4103 - Recommender Systems
Date: October 2025

This module is a lightweight demo used during development and testing. It is
not required for automated evaluation but is handy when exploring recommendations.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import List
import traceback
import os

from .data import load_ratings, build_user_item_matrix
from .model import UserCF

MODEL: UserCF = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global MODEL
    data_dir = os.environ.get("ML_DATA_DIR", "data/ml-latest-small")
    try:
        ratings = load_ratings(data_dir)
        user_item = build_user_item_matrix(ratings)
        model = UserCF(k=20)
        model.fit(user_item)
        MODEL = model
        print("Model loaded. Users:", len(user_item))
    except Exception:
        print("Model not loaded at startup. You may need to download the dataset first.")
        traceback.print_exc()
    
    yield
    
    # Shutdown (cleanup if needed)
    MODEL = None


app = FastAPI(title="Movie Recommender - UserCF", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10) -> List[int]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        recs = MODEL.recommend(user_id, n=n)
        return recs
    except KeyError:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
