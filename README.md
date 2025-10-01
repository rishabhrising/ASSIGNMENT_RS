# Movie Recommender System Using Collaborative Filtering

## Overview
This project demonstrates a movie recommendation system using user-based collaborative filtering. I built this as part of my AI4103 Recommender Systems course to understand how recommendation algorithms work in practice.

## What I Learned
Through this project, I explored:
- How collaborative filtering finds similar users based on rating patterns
- The difference between cosine similarity and Pearson correlation
- Why mean-centering ratings can improve recommendations
- How to evaluate recommender systems using multiple metrics
- The challenges of cold start problems and data sparsity

## Project Structure
```
├── src/recommender/          # Core implementation
│   ├── data.py              # Data loading and preprocessing
│   ├── model.py             # UserCF algorithm implementation
│   └── api.py               # Simple web API for testing
├── scripts/                 # Evaluation and utilities
│   └── evaluate_movielens.py # Comprehensive evaluation script
├── FINAL_REPORT.pdf         # Detailed analysis and results
└── requirements.txt         # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Basic understanding of machine learning concepts

### Installation
1. Clone this repository and navigate to it
2. Set up a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Demo
Run the evaluation on MovieLens dataset:
```bash
python scripts/evaluate_movielens.py --k 20 --krec 10 --metric cosine
```

This will:
1. Download the MovieLens small dataset automatically
2. Split data into train/test sets
3. Train the collaborative filtering model
4. Report RMSE and ranking metrics

### Try Different Configurations
```bash
# Use Pearson correlation with mean centering
python scripts/evaluate_movielens.py --metric pearson --mean-center

# Add baseline predictors
python scripts/evaluate_movielens.py --baseline --k 30
```

### Interactive Testing (Optional)
Start the web API to explore recommendations:
```bash
uvicorn src.recommender.api:app --reload --port 8000
```
Then visit `http://localhost:8000/recommend/1` to see recommendations for user 1.

## Implementation Details
The core algorithm implements user-based collaborative filtering:
1. Computes user-user similarity using cosine or Pearson correlation
2. For each prediction, finds K most similar users who rated the item
3. Predicts rating as weighted average of similar users' ratings
4. Optionally applies mean centering and baseline predictors

See `FINAL_REPORT.pdf` for detailed methodology and results analysis.

## Results Summary
On MovieLens ml-latest-small dataset:
- RMSE: 3.66 (rating prediction accuracy)
- HitRate@10: 10.16% (recommendation relevance)
- Precision@10: 1.02%
- Recall@10: 10.16%

## Author
Rishabh Rising - AI4103 Recommender Systems Course
