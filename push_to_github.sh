#!/bin/bash
# Script to push code to GitHub

# Ensure we're in the correct directory
cd "$(dirname "$0")"

echo "Preparing to push SkinSight to GitHub..."

# Configure git (uncomment and replace with your details)
# git config user.name "Your Name"
# git config user.email "your.email@example.com"

# Commit changes
git commit -m "Initial commit of SkinSight - Multimodal Skin Lesion Diagnosis"

# Add remote repository (GitHub repo URL)
git remote add origin https://github.com/abhinav30219/SkinSight.git

echo "Ready to push to GitHub. You have two options:"

echo "1. Push using HTTPS (you'll need to enter your GitHub credentials):"
echo "   git push -u origin main"

echo "2. Push using SSH (if you have SSH keys set up):"
echo "   git push -u origin main"

echo ""
echo "After pushing, your code will be available at: https://github.com/abhinav30219/SkinSight"
echo ""
echo "To execute this push automatically, run:"
echo "   git push -u origin main"
