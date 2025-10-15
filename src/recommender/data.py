"""Data loading helpers for MovieLens (ratings loader and pivot builder).

Author: rishabhrising
Course: AI4103 - Recommender Systems
Date: October 2025
"""

import os
import pandas as pd


def load_ratings(data_dir: str = "data/ml-latest-small") -> pd.DataFrame:
    """Load ratings.csv from MovieLens extracted folder. Returns DataFrame with columns userId,itemId,rating,timestamp"""
    ratings_path = os.path.join(data_dir, "ratings.csv")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings.csv not found in {data_dir}. Run scripts/download_movielens.py first.")
    df = pd.read_csv(ratings_path)
    return df


def build_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Return user-item pivot table (users as rows, items as columns) with ratings and NaN for missing."""
    pivot = ratings_df.pivot(index="userId", columns="movieId", values="rating")
    return pivot
