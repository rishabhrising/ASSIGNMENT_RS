"""Data loading helpers for MovieLens dataset.

Author: rishabhrising
Course: AI4103 - Recommender Systems
Date: October 2025
"""

import os
import pandas as pd


def load_ratings(data_dir: str = "data/ml-latest-small") -> pd.DataFrame:
    """Load ratings.csv from MovieLens extracted folder."""
    ratings_path = os.path.join(data_dir, "ratings.csv")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings.csv not found in {data_dir}")
    return pd.read_csv(ratings_path)


def build_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Build user-item matrix from ratings DataFrame."""
    return ratings_df.pivot(index="userId", columns="movieId", values="rating")
