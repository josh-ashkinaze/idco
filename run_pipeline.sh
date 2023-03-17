#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]
  then
    echo "Please provide the number of comments as an argument."
    exit 1
fi

# Get the number of comments from the argument
num_comments=$1

# Navigate to the 'scripts/get_data' directory
cd scripts/get_data

# Run the 'get_comments.py' script
echo "Running get_comments.py with $num_comments comments per subreddit..."
python3 get_comments.py $num_comments

# Run the 'preprocess_comments.py' script
echo "Running preprocess_comments.py..."
python3 preprocess_comments.py

# Train w2v
cd ../nn
python3 train_nn_.py

echo "Pipeline complete."
