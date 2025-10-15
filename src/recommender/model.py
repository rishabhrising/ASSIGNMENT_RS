"""User-Based Collaborative Filtering Implementation

Author: Rishabh Rising
Course: AI4103 - Recommender Systems
Date: October 2024

This module implements user-based collaborative filtering for movie recommendations.
The algorithm finds users with similar rating patterns and uses their preferences
to predict ratings for unseen items.

Key Features:
- Support for both cosine similarity and Pearson correlation
- Optional mean-centering to remove user rating bias
- Baseline predictor integration for improved accuracy
- Comprehensive error handling for edge cases
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import warnings


class UserCF:
    """User-Based Collaborative Filtering Recommender
    
    This class implements a user-based collaborative filtering algorithm that:
    1. Computes user-user similarity based on rating patterns
    2. Predicts ratings using weighted averages from similar users
    3. Recommends items by ranking predicted ratings
    
    Args:
        k (int): Number of similar users to consider for predictions
        metric (str): Similarity metric - 'cosine' or 'pearson'
        mean_center (bool): Whether to center ratings by user mean
        use_baseline (bool): Whether to use baseline predictors
    """
    
    def __init__(self, k: int = 20, metric: str = "cosine", 
                 mean_center: bool = False, use_baseline: bool = False):
        # Validate input parameters
        if k <= 0:
            raise ValueError("k must be positive")
        if metric not in ['cosine', 'pearson']:
            raise ValueError("metric must be 'cosine' or 'pearson'")
            
        # Algorithm configuration
        self.k = k
        self.metric = metric
        self.mean_center = mean_center
        self.use_baseline = use_baseline
        
        # Model state (populated during fit)
        self.user_item = None           # User-item matrix with ratings
        self.user_index = None          # Mapping from user_id to matrix index
        self.sim_matrix = None          # Precomputed similarity matrix
        self.user_means = None          # Mean rating per user
        self.global_mean = None         # Overall mean rating
        self.user_bias = None           # User bias terms for baseline
        self.item_bias = None           # Item bias terms for baseline
        
        self._is_fitted = False         # Track model state

    def fit(self, user_item: pd.DataFrame) -> 'UserCF':
        """Train the collaborative filtering model
        
        This method processes the user-item rating matrix and computes the necessary
        components for making predictions:
        1. User rating statistics (means, biases)
        2. User-user similarity matrix
        3. Baseline predictors (if enabled)
        
        Args:
            user_item (pd.DataFrame): User-item rating matrix with users as rows,
                                    items as columns, and ratings as values.
                                    Missing ratings should be NaN.
        
        Returns:
            UserCF: Returns self for method chaining
            
        Raises:
            ValueError: If input data is invalid or empty
        """
        # Input validation
        if user_item.empty:
            raise ValueError("Input user_item matrix cannot be empty")
        if not isinstance(user_item, pd.DataFrame):
            raise ValueError("user_item must be a pandas DataFrame")
            
        print(f"Training UserCF with {len(user_item)} users and {len(user_item.columns)} items...")
        
        # Step 1: Prepare the rating matrix
        ui_original = user_item.copy()
        
        # Calculate user statistics before any transformations
        # Only consider actual ratings (non-NaN) for computing means
        self.user_means = ui_original.mean(axis=1)  # Mean rating per user
        self.user_index = list(user_item.index)     # User ID to index mapping
        
        # Step 2: Apply mean centering if requested
        if self.mean_center:
            # Subtract each user's mean from their ratings
            # This removes user-specific rating bias
            ui_processed = ui_original.sub(self.user_means, axis=0)
            print("Applied mean-centering to remove user rating bias")
        else:
            ui_processed = ui_original.copy()
            
        # Replace NaN with 0 for similarity computation
        # Note: This is a simplification - more sophisticated approaches exist
        self.user_item = ui_processed.fillna(0)
        
        # Step 3: Compute baseline predictors if requested
        if self.use_baseline:
            self._compute_baseline_terms(ui_original)
            print("Computed baseline predictors (global/user/item biases)")
            
        # Step 4: Compute user-user similarity matrix
        self._compute_similarity_matrix()
        print(f"Computed {self.metric} similarity matrix ({len(self.user_index)}x{len(self.user_index)})")
        
        self._is_fitted = True
        return self
    
    def _compute_baseline_terms(self, ui_original: pd.DataFrame) -> None:
        """Compute baseline predictor components"""
        # Global mean rating across all users and items
        self.global_mean = ui_original.stack().mean()
        
        # Item means (average rating per item)
        item_means = ui_original.mean(axis=0)
        
        # Compute bias terms: how much each user/item deviates from global mean
        self.user_bias = (self.user_means - self.global_mean).to_dict()
        self.item_bias = (item_means - self.global_mean).to_dict()
    
    def _compute_similarity_matrix(self) -> None:
        """Compute user-user similarity matrix based on rating patterns"""
        if self.metric == "cosine":
            # Cosine similarity: measures angle between rating vectors
            self.sim_matrix = cosine_similarity(self.user_item.values)
            
        elif self.metric == "pearson":
            # Pearson correlation: measures linear relationship between ratings
            # For Pearson, we need to center the data (subtract mean)
            data_for_pearson = self.user_item.copy()
            
            if not self.mean_center:
                # If we didn't mean-center earlier, do it now for Pearson
                user_means_expanded = self.user_means.values[:, np.newaxis]
                data_for_pearson = data_for_pearson.values - user_means_expanded
                # Handle any remaining NaN values
                data_for_pearson = np.nan_to_num(data_for_pearson, nan=0.0)
            else:
                data_for_pearson = data_for_pearson.values
                
            # Compute cosine similarity on centered data = Pearson correlation
            self.sim_matrix = cosine_similarity(data_for_pearson)
        else:
            # This should never happen due to __init__ validation
            raise ValueError(f"Unknown similarity metric: {self.metric}")

    def _user_idx(self, user_id: int) -> int:
        """Get the matrix index for a given user ID"""
        try:
            return self.user_index.index(user_id)
        except ValueError:
            raise KeyError(f"User {user_id} not found in training data")

    def predict(self, user_id: int, item_id) -> float:
        """Predict rating for a specific user-item pair
        
        This is the core prediction algorithm that:
        1. Finds the K most similar users who have rated the target item
        2. Computes a weighted average of their ratings
        3. Optionally applies mean-centering and baseline corrections
        
        Args:
            user_id (int): ID of the user for whom to predict
            item_id: ID of the item to predict rating for
            
        Returns:
            float: Predicted rating, or NaN if prediction cannot be made
            
        Raises:
            RuntimeError: If model hasn't been fitted yet
            KeyError: If user_id is not in training data
        """
        # Ensure model is trained
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        # Get user's index in the matrix
        uidx = self._user_idx(user_id)
        
        # Check if item exists in our data
        if item_id not in self.user_item.columns:
            warnings.warn(f"Item {item_id} not in training data")
            return float(np.nan)
            
        # Step 1: Get similarity scores for this user with all other users
        user_similarities = self.sim_matrix[uidx].copy()
        
        # Step 2: Get all users' ratings for this item
        item_ratings = self.user_item[item_id].values
        
        # Step 3: Exclude the target user from being their own neighbor
        user_similarities[uidx] = 0.0
        
        # Step 4: Find the K most similar users
        top_k_indices = np.argsort(user_similarities)[-self.k:]
        top_k_sims = user_similarities[top_k_indices]
        top_k_ratings = item_ratings[top_k_indices]
        
        # Step 5: Handle case where no similar users have meaningful similarity
        if np.sum(top_k_sims) == 0:
            return float(np.nan)
            
        # Step 6: Compute prediction based on configuration
        if self.use_baseline:
            prediction = self._predict_with_baseline(user_id, item_id, top_k_indices, top_k_sims, top_k_ratings)
        else:
            prediction = self._predict_basic(user_id, top_k_sims, top_k_ratings)
            
        return float(prediction)
    
    def _predict_basic(self, user_id: int, similarities: np.ndarray, ratings: np.ndarray) -> float:
        """Basic collaborative filtering prediction"""
        # Weighted average of similar users' ratings
        prediction = np.dot(similarities, ratings) / np.sum(similarities)
        
        # Add back user's mean if we mean-centered during training
        if self.mean_center:
            user_mean = self.user_means.get(user_id, 0.0)
            prediction += user_mean
            
        return prediction
    
    def _predict_with_baseline(self, user_id: int, item_id, neighbor_indices: np.ndarray, 
                              similarities: np.ndarray, ratings: np.ndarray) -> float:
        """Prediction using baseline predictors (user/item biases)"""
        # Compute baseline prediction for target user-item pair
        global_mean = self.global_mean or self.user_means.mean()
        user_bias = self.user_bias.get(user_id, 0.0)
        item_bias = self.item_bias.get(item_id, 0.0)
        baseline_prediction = global_mean + user_bias + item_bias
        
        # For each neighbor, compute their baseline and use (actual - baseline)
        neighbor_user_ids = [self.user_index[i] for i in neighbor_indices]
        neighbor_baselines = []
        
        for neighbor_id in neighbor_user_ids:
            neighbor_user_bias = self.user_bias.get(neighbor_id, 0.0)
            neighbor_item_bias = self.item_bias.get(item_id, 0.0)
            neighbor_baseline = global_mean + neighbor_user_bias + neighbor_item_bias
            neighbor_baselines.append(neighbor_baseline)
            
        neighbor_baselines = np.array(neighbor_baselines)
        
        # Use rating deviations from baseline instead of raw ratings
        rating_deviations = ratings - neighbor_baselines
        
        # Weighted average of deviations
        deviation_prediction = np.dot(similarities, rating_deviations) / np.sum(similarities)
        
        # Final prediction = baseline + weighted deviation
        final_prediction = baseline_prediction + deviation_prediction
        
        return final_prediction

    def recommend(self, user_id: int, n: int = 10) -> List:
        """Generate top-N item recommendations for a user
        
        This method predicts ratings for all items the user hasn't rated yet,
        then returns the N items with highest predicted ratings.
        
        Args:
            user_id (int): ID of the user to generate recommendations for
            n (int): Number of recommendations to return
            
        Returns:
            List: List of item IDs ranked by predicted rating (highest first)
            
        Raises:
            RuntimeError: If model hasn't been fitted yet
            KeyError: If user_id is not in training data
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making recommendations")
            
        # Get user's index and current ratings
        uidx = self._user_idx(user_id)
        user_ratings = self.user_item.iloc[uidx]
        
        # Find items the user hasn't rated (rating = 0 in our filled matrix)
        unrated_items = user_ratings[user_ratings == 0].index.tolist()
        
        if not unrated_items:
            warnings.warn(f"User {user_id} has rated all available items")
            return []
            
        # Predict ratings for all unrated items
        item_scores = []
        for item in unrated_items:
            try:
                predicted_rating = self.predict(user_id, item)
                # Only include items where we could make a prediction
                if not pd.isna(predicted_rating):
                    item_scores.append((item, predicted_rating))
            except Exception as e:
                # Skip items that cause prediction errors
                continue
                
        # Sort by predicted rating (descending) and return top N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [item for item, score in item_scores[:n]]
        
        return recommendations
