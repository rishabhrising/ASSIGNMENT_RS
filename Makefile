# Movie Recommender System - Development Tasks
# Author: Rishabh Rising

.PHONY: help install test demo evaluate clean

# Default target
help:
	@echo "Movie Recommender System - Available Commands:"
	@echo ""
	@echo "  install    - Install Python dependencies"
	@echo "  demo       - Run basic usage demonstration"
	@echo "  evaluate   - Run full evaluation on MovieLens dataset"
	@echo "  test       - Run simple functionality tests"
	@echo "  clean      - Clean up generated files"
	@echo "  api        - Start the web API server for interactive testing"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make demo"
	@echo "  make evaluate ARGS='--metric pearson --mean-center'"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Installation complete!"

# Run the basic demonstration
demo:
	@echo "Running basic usage demonstration..."
	python example_usage.py

# Run full evaluation on MovieLens dataset
evaluate:
	@echo "Running evaluation on MovieLens dataset..."
	python scripts/evaluate_movielens.py $(ARGS)

# Run simple tests to verify functionality
test:
	@echo "Running basic functionality tests..."
	@echo "Testing imports..."
	python -c "from src.recommender.model import UserCF; print('✓ Model import successful')"
	python -c "from src.recommender.data import load_ratings; print('✓ Data module import successful')"
	@echo "Testing basic model creation..."
	python -c "from src.recommender.model import UserCF; model = UserCF(); print('✓ Model creation successful')"
	@echo "All basic tests passed!"

# Start API server for interactive testing  
api:
	@echo "Starting web API server..."
	@echo "Visit http://localhost:8000/health to check status"
	@echo "Visit http://localhost:8000/recommend/1 for sample recommendations"
	uvicorn src.recommender.api:app --reload --port 8000

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	rm -rf data/ml-latest-small* 2>/dev/null || true
	@echo "Cleanup complete!"

# Quick evaluation with different configurations
eval-cosine:
	python scripts/evaluate_movielens.py --metric cosine --k 20

eval-pearson:
	python scripts/evaluate_movielens.py --metric pearson --mean-center --k 20

eval-baseline:
	python scripts/evaluate_movielens.py --metric pearson --mean-center --baseline --k 20