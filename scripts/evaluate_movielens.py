"""Evaluate UserCF on MovieLens (ml-latest-small).

Author: rishabhrising
Course: AI4103 - Recommender Systems
Date: October 2025

Usage:
    python scripts/evaluate_movielens.py --data-dir data --k 20 --krec 10

This script will:
 - download ml-latest-small into data/ if missing
 - do a per-user single holdout (one rating per user reserved for test)
 - train UserCF on the remaining ratings
 - compute RMSE on test ratings and HitRate@K (how often the held-out item is in top-K)
"""
import argparse
import os
import random
import sys
import math

import pandas as pd
import numpy as np

# ensure src/ and project root are importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from recommender.data import load_ratings, build_user_item_matrix
from recommender.model import UserCF


def download_if_missing(dest='data'):
    extracted = os.path.join(dest, 'ml-latest-small')
    ratings_path = os.path.join(extracted, 'ratings.csv')
    if os.path.exists(ratings_path):
        print('Found existing MovieLens data at', extracted)
        return extracted
    print('Downloading ml-latest-small...')
    # inline download to avoid external script dependency
    ML_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "ml-latest-small.zip")
    try:
        from urllib.request import urlretrieve
        urlretrieve(ML_SMALL_URL, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dest)
        return extracted
    except Exception as e:
        raise RuntimeError(f"Failed to download or extract MovieLens: {e}")


def train_test_holdout(ratings_df, min_ratings=5, seed=42):
    """Do a simple per-user holdout: reserve one rating per active user for testing.

    Users with fewer than `min_ratings` keep all ratings in train.
    """
    random.seed(seed)
    test_rows = []
    train_rows = []
    grouped = ratings_df.groupby('userId')
    for uid, group in grouped:
        if len(group) < min_ratings:
            # keep all in train for low-activity users
            train_rows.append(group)
            continue
        # pick one random index as test
        test_idx = random.choice(list(group.index))
        test_rows.append(group.loc[[test_idx]])
        train_rows.append(group.drop(index=test_idx))
    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame(columns=ratings_df.columns)
    return train_df, test_df


def precision_at_k(recommended, relevant, k):
    if not recommended:
        return 0.0
    recommended_k = recommended[:k]
    rel_set = set(relevant)
    hits = sum(1 for r in recommended_k if r in rel_set)
    return hits / len(recommended_k)


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    rel_set = set(relevant)
    hits = sum(1 for r in recommended_k if r in rel_set)
    return hits / len(rel_set)


def apk(recommended, relevant, k):
    """Average precision at k for a single user."""
    if not recommended:
        return 0.0
    recommended_k = recommended[:k]
    score = 0.0
    hits = 0.0
    rel_set = set(relevant)
    for i, p in enumerate(recommended_k, start=1):
        if p in rel_set:
            hits += 1.0
            score += hits / i
    return score / min(len(rel_set), k) if len(rel_set) > 0 else 0.0


def mapk(all_recommended, all_relevant, k):
    apks = []
    for recs, rel in zip(all_recommended, all_relevant):
        apks.append(apk(recs, rel, k))
    return float(np.mean(apks)) if apks else 0.0


def evaluate(train_df, test_df, k_neighbors=20, k_rec=10, mean_center=False, metric='cosine', baseline=False):
    user_item_train = build_user_item_matrix(train_df)
    model = UserCF(k=k_neighbors, metric=metric, mean_center=mean_center, use_baseline=baseline)
    model.fit(user_item_train)

    # RMSE on test set
    preds = []
    actuals = []
    hits = 0
    total = 0
    for _, row in test_df.iterrows():
        uid = row['userId']
        mid = row['movieId']
        true = row['rating']
        try:
            pred = model.predict(uid, mid)
        except KeyError:
            # user not in train (shouldn't happen with our split) -> skip
            continue
        if math.isnan(pred):
            # fallback: user's mean in train
            try:
                uidx = model._user_idx(uid)
                pred = model.user_item.iloc[uidx][model.user_item.iloc[uidx] > 0].mean()
                if math.isnan(pred):
                    # global mean
                    pred = train_df['rating'].mean()
            except Exception:
                pred = train_df['rating'].mean()
        preds.append(pred)
        actuals.append(true)
    preds = np.array(preds)
    actuals = np.array(actuals)
    rmse = np.sqrt(np.mean((preds - actuals) ** 2)) if len(preds) > 0 else float('nan')

    # Additional ranking metrics: Precision@K, Recall@K, MAP@K
    users_with_test = test_df['userId'].unique()
    all_recommended = []
    all_relevant = []
    for uid in users_with_test:
        user_tests = test_df[test_df['userId'] == uid]
        test_item = int(user_tests.iloc[0]['movieId'])
        try:
            recs = model.recommend(uid, n=k_rec)
        except KeyError:
            continue
        total += 1
        if test_item in recs:
            hits += 1
        all_recommended.append(recs)
        all_relevant.append([test_item])
    hr = hits / total if total > 0 else float('nan')
    precision = np.mean([precision_at_k(r, rel, k_rec) for r, rel in zip(all_recommended, all_relevant)]) if all_recommended else float('nan')
    recall = np.mean([recall_at_k(r, rel, k_rec) for r, rel in zip(all_recommended, all_relevant)]) if all_recommended else float('nan')
    mapk_score = mapk(all_recommended, all_relevant, k_rec) if all_recommended else float('nan')
    return rmse, hr, precision, recall, mapk_score, len(user_item_train), len(test_df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Destination for MovieLens data')
    parser.add_argument('--k', type=int, default=20, help='Number of neighbors for UserCF')
    parser.add_argument('--krec', type=int, default=10, help='k for HitRate@k')
    parser.add_argument('--metric', choices=['cosine', 'pearson'], default='cosine', help='Similarity metric')
    parser.add_argument('--mean-center', action='store_true', help='Mean-center user ratings before computing similarity')
    parser.add_argument('--baseline', action='store_true', help='Use baseline predictors (global/user/item biases)')
    args = parser.parse_args()

    data_extracted = download_if_missing(args.data_dir)
    ratings = load_ratings(data_extracted)
    print('Total ratings:', len(ratings))
    train_df, test_df = train_test_holdout(ratings)
    print('Train size:', len(train_df), 'Test size:', len(test_df))

    rmse, hr, precision, recall, mapk_score, n_users, n_tests = evaluate(
        train_df,
        test_df,
        k_neighbors=args.k,
        k_rec=args.krec,
        mean_center=args.mean_center,
        metric=args.metric,
        baseline=args.baseline,
    )
    print('Configuration: metric=', args.metric, 'mean_center=', args.mean_center, 'baseline=', args.baseline)
    print('Evaluation results:')
    print(f'  Users in train: {n_users}')
    print(f'  Test events: {n_tests}')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  HitRate@{args.krec}: {hr:.4f}')
    print(f'  Precision@{args.krec}: {precision:.4f}')
    print(f'  Recall@{args.krec}: {recall:.4f}')
    print(f'  MAP@{args.krec}: {mapk_score:.4f}')


if __name__ == '__main__':
    main()
