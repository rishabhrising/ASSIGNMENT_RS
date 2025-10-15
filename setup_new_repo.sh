#!/bin/bash
# Script to prepare repository for new GitHub upload
# Author: Setup script for assignment_RS repository

set -e  # Exit on any error

echo "🔧 Preparing repository for new GitHub upload..."

# Remove old git history
echo "📝 Removing old git history..."
rm -rf .git

# Clean up any temporary files
echo "🧹 Cleaning up temporary files..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true

# Create/update .gitignore
echo "📋 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# macOS
.DS_Store

# Data files
data/
*.db
*.sqlite

# Logs
*.log

# API keys and secrets
.env.*
secrets.txt
EOF

# Initialize new git repository
echo "🚀 Initializing new git repository..."
git init

# Add all files
echo "📦 Adding files to git..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Movie Recommender System using User-Based Collaborative Filtering

Implementation of collaborative filtering algorithm for AI4103 course assignment.
Features cosine/Pearson similarity, mean-centering, baseline predictors, and 
comprehensive evaluation on MovieLens dataset."

echo ""
echo "✅ Repository prepared successfully!"
echo ""
echo "🔗 Next steps:"
echo "1. Go to https://github.com/new"
echo "2. Create a new repository named 'assignment_RS' (or similar)"
echo "3. Don't initialize with README, .gitignore, or license"
echo "4. Copy the commands GitHub provides, which will be something like:"
echo ""
echo "   git remote add origin https://github.com/yourusername/assignment_RS.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "📊 Repository stats:"
echo "   - $(find . -name "*.py" | wc -l | tr -d ' ') Python files"
echo "   - $(find . -name "*.md" | wc -l | tr -d ' ') Markdown files" 
echo "   - $(git log --oneline | wc -l | tr -d ' ') commit ready to push"
echo ""
echo "🎯 Ready for submission!"