#!/usr/bin/env python3
"""Simple example demonstrating the movie recommender system

This script shows how to:
1. Load a small sample dataset
2. Train the collaborative filtering model
3. Make predictions and recommendations
4. Compare different similarity metrics

Author: Rishabh Rising
Course: AI4103 - Recommender Systems
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from recommender.model import UserCF


def create_sample_data():
    """Create a small synthetic dataset for demonstration"""
    
    # Sample user-item rating matrix
    # Users are rows, movies are columns
    # Ratings on 1-5 scale, NaN means no rating
    sample_ratings = {
        'Movie_1': [5.0, 3.0, np.nan, 1.0, np.nan],
        'Movie_2': [4.0, np.nan, np.nan, 1.0, np.nan], 
        'Movie_3': [np.nan, 1.0, np.nan, 5.0, 4.0],
        'Movie_4': [np.nan, 2.0, 4.0, 4.0, 3.0],
        'Movie_5': [np.nan, 3.0, 5.0, 3.0, 5.0],
    }
    
    # Create DataFrame with user IDs as index
    user_ids = [101, 102, 103, 104, 105]
    df = pd.DataFrame(sample_ratings, index=user_ids)
    
    return df


def demonstrate_basic_usage():
    """Show basic model training and prediction"""
    print("=" * 60)
    print("BASIC USAGE DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    print("1. Creating sample rating data...")
    ratings_matrix = create_sample_data()
    print("Sample ratings matrix:")
    print(ratings_matrix)
    print()
    
    # Train the model
    print("2. Training collaborative filtering model...")
    model = UserCF(k=3, metric='cosine')
    model.fit(ratings_matrix)
    print()
    
    # Make a prediction
    print("3. Making rating predictions...")
    user_id = 101
    movie_id = 'Movie_3'  # User 101 hasn't rated Movie_3
    
    try:
        predicted_rating = model.predict(user_id, movie_id)
        print(f"Predicted rating for User {user_id} on {movie_id}: {predicted_rating:.2f}")
    except Exception as e:
        print(f"Prediction failed: {e}")
    print()
    
    # Get recommendations
    print("4. Getting top recommendations...")
    try:
        recommendations = model.recommend(user_id, n=3)
        print(f"Top 3 movie recommendations for User {user_id}: {recommendations}")
    except Exception as e:
        print(f"Recommendation failed: {e}")
    print()


def compare_similarity_metrics():
    """Compare cosine vs Pearson similarity"""
    print("=" * 60) 
    print("SIMILARITY METRIC COMPARISON")
    print("=" * 60)
    
    ratings_matrix = create_sample_data()
    user_id = 102
    movie_id = 'Movie_1'
    
    metrics = ['cosine', 'pearson']
    
    for metric in metrics:
        print(f"\nUsing {metric.upper()} similarity:")
        
        # Train model with current metric
        model = UserCF(k=3, metric=metric, mean_center=(metric=='pearson'))
        model.fit(ratings_matrix)
        
        # Make prediction
        try:
            pred = model.predict(user_id, movie_id)
            print(f"  Predicted rating for User {user_id} on {movie_id}: {pred:.2f}")
            
            # Get recommendations
            recs = model.recommend(user_id, n=2)
            print(f"  Top 2 recommendations: {recs}")
            
        except Exception as e:
            print(f"  Error: {e}")


def demonstrate_advanced_features():
    """Show mean centering and baseline predictors"""
    print("=" * 60)
    print("ADVANCED FEATURES DEMONSTRATION") 
    print("=" * 60)
    
    ratings_matrix = create_sample_data()
    user_id = 103
    movie_id = 'Movie_2'
    
    configurations = [
        {'name': 'Basic', 'params': {}},
        {'name': 'Mean-Centered', 'params': {'mean_center': True}},
        {'name': 'With Baseline', 'params': {'use_baseline': True}},
        {'name': 'Full Featured', 'params': {'mean_center': True, 'use_baseline': True}},
    ]
    
    for config in configurations:
        print(f"\n{config['name']} Configuration:")
        
        try:
            # Create and train model
            model = UserCF(k=3, metric='pearson', **config['params'])
            model.fit(ratings_matrix)
            
            # Make prediction
            pred = model.predict(user_id, movie_id)
            print(f"  Predicted rating: {pred:.2f}")
            
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Run all demonstrations"""
    print("Movie Recommender System - Usage Examples")
    print("Author: Rishabh Rising")
    print()
    
    try:
        demonstrate_basic_usage()
        compare_similarity_metrics() 
        demonstrate_advanced_features()
        
        print("=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("For real-world usage with MovieLens data, run:")
        print("  python scripts/evaluate_movielens.py")
        print()
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()