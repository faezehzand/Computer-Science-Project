Hybrid Recommendation System
Project Overview
This project is a Hybrid Recommendation System for an educational platform. It combines Collaborative Filtering and Content-Based Filtering to recommend lessons to users based on their interaction data and lesson attributes. The system aims to enhance the user experience by providing personalized lesson recommendations.

Key Components:
Collaborative Filtering: Recommends lessons by identifying similar users based on their past ratings.
Content-Based Filtering: Recommends lessons based on the similarity of lesson attributes (e.g., topics, difficulty).
Hybrid Model: Combines both collaborative and content-based methods for improved recommendations.

Installation
1. Clone the Repository
To start using the project, first clone the repository to your local machine:

git clone https://github.com/faezehzand/Computer-Science-Project.git
cd Computer-Science-Project

2. Set Up a Virtual Environment (Optional)
It’s recommended to create a virtual environment to manage dependencies:

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate  # For Windows

3. Install Dependencies
pip install pandas numpy scikit-learn matplotlib

Run Instructions
Here’s how to run the different parts of the project.

1. Generate Simulated Dataset
Generate a dataset of user interactions and lesson attributes using this script:
python generate_simulated_dataset.py
2. Preprocess Data
Preprocess the data for analysis and model training:
python preprocessing_data.py
3. Exploratory Data Analysis (Optional)
To visualize and understand the dataset:
python exploratory_data_analysis.py
4. Run the Hybrid Recommendation System
Generate lesson recommendations using the hybrid recommendation model:
python make_prediction.py


Function Documentation
Below is a summary of the main functions in the project:

generate_simulated_dataset.py
Description: Generates a simulated dataset with user interactions and lesson attributes.
Functions:
  generate_interactions(): Simulates interactions between users and lessons.
  generate_lessons(): Creates a dataset with lesson metadata (e.g., topics, difficulty).
  
preprocessing_data.py
Description: Preprocesses the dataset for analysis.
Functions:
  preprocess_interactions(): Cleans and processes user-lesson interactions.
  preprocess_lessons(): Prepares lesson metadata for the content-based filtering model.
  
make_prediction.py
Description: Implements collaborative filtering, content-based filtering, and hybrid recommendations.
Functions:
  recommend_collaborative(user_id, num_recommendations=5): Generates lesson recommendations using collaborative filtering.
  recommend_content_based(lesson_id, num_recommendations=5): Recommends lessons based on their similarity to a given lesson.
  hybrid_recommendations(user_id, num_recommendations=5): Combines both collaborative and content-based methods for hybrid recommendations.
  
evaluate_models.py
Description: Evaluates the performance of the recommendation models.
Functions:
  evaluate_model(y_true, y_pred, model_name): Computes precision, recall, F1-score, and MSE for the models.

Special thanks for your guidance and valuable feedback throughout the project.

