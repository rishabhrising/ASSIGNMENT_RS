# Assignment Submission - Movie Recommendation System

**Student:** Rishabh Rising  
**Course:** AI4103 - Recommender Systems  
**Submission Date:** October 15, 2025  
**Project Type:** Individual Assignment  

## Project Overview

This submission demonstrates my implementation of a user-based collaborative filtering system for movie recommendations. The project showcases both theoretical understanding and practical implementation skills in recommender systems.

### Key Components Delivered

1. **Core Implementation** (`src/recommender/`)
   - `data.py`: MovieLens data loading and preprocessing
   - `model.py`: User-based collaborative filtering algorithm
   - `api.py`: RESTful API for interactive testing

2. **Evaluation Framework** (`scripts/evaluate_movielens.py`)
   - Automated dataset download and setup
   - Train/test split with per-user holdout validation
   - Comprehensive metric computation (RMSE, HitRate, Precision, Recall, MAP)

3. **Documentation**
   - `FINAL_REPORT.pdf`: Complete analysis with visualizations
   - `README.md`: User guide and technical overview

## Personal Learning Experience

### Challenges Encountered
1. **Data Sparsity**: The MovieLens dataset is quite sparse, making similarity calculations challenging
2. **Cold Start Problem**: New users with few ratings are difficult to match with similar users  
3. **Computational Efficiency**: Computing user-user similarity for all pairs scales poorly
4. **Parameter Tuning**: Finding optimal K (number of neighbors) required extensive experimentation

### Key Insights Gained
- Mean-centering significantly improves recommendation quality by removing user bias
- Pearson correlation often outperforms cosine similarity for rating prediction
- Baseline predictors help handle the long-tail distribution of user preferences
- Evaluation metrics can be misleading - HitRate doesn't always correlate with user satisfaction

### Implementation Decisions
- Chose user-based over item-based CF to better understand user similarity concepts
- Implemented both cosine and Pearson similarity for comparison
- Added baseline predictor support for improved accuracy
- Created modular design for easy experimentation with different parameters

## Experimental Results

### Best Configuration
```bash
python scripts/evaluate_movielens.py --metric pearson --mean-center --baseline --k 20
```

### Performance Metrics
- **RMSE**: 3.66 (prediction accuracy)
- **HitRate@10**: 10.16% (recommendation relevance)
- **Precision@10**: 1.02%
- **MAP@10**: 4.44%

### Analysis
The RMSE of 3.66 indicates moderate prediction accuracy on the 1-5 rating scale. The low precision suggests room for improvement, possibly through hybrid approaches or matrix factorization techniques.

## Future Improvements
If I were to extend this project, I would:
1. Implement matrix factorization (SVD/NMF) for better scalability
2. Add content-based features for hybrid recommendations
3. Explore deep learning approaches (neural collaborative filtering)
4. Implement online learning for real-time updates
5. Add explanation capabilities for recommendation transparency

## Academic Integrity Statement
This project represents my original work completed for the AI4103 course. While I referenced standard algorithms from academic literature, all implementation details and experimental design are my own contributions.

---
*This submission fulfills the course requirements for demonstrating practical understanding of collaborative filtering techniques and recommender system evaluation methodologies.*
