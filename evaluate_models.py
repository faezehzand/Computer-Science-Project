import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from make_prediction import recommend_collaborative, recommend_content_based, hybrid_recommendations

# Load the preprocessed interactions data
df_interactions = pd.read_csv("preprocessed_interactions.csv")

# Split the dataset into training and test sets (80% train, 20% test)
train_data, test_data = train_test_split(df_interactions, test_size=0.2, random_state=42)

# Ground truth (actual ratings) from the test set
y_true = test_data['rating'].values


def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate the model using precision, recall, F1-score, and mean squared error (MSE).

    Parameters:
    y_true (array): The actual ratings from the test set.
    y_pred (array): The predicted ratings for the test set.
    model_name (str): The name of the model being evaluated (for display).

    Returns:
    None: Prints the precision, recall, F1-score, and MSE for the model.
    """
    y_pred = np.array(y_pred)

    # Convert predictions and ground truth into binary format for precision, recall, and F1 calculations
    y_true_binary = (y_true > 0).astype(int)  # Binary: 1 for rated items, 0 otherwise
    y_pred_binary = (y_pred > 0).astype(int)  # Binary: 1 for predicted rated items, 0 otherwise

    # Compute evaluation metrics
    precision = precision_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
    mse = mean_squared_error(y_true, y_pred)

    # Print out the performance metrics for the model
    print(f"Performance of {model_name}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"MSE: {mse:.4f}")
    print("-" * 50)


# Specify a test user ID for the evaluation
user_id = 1

# 1. Collaborative Filtering Evaluation
collab_recommendations = recommend_collaborative(user_id, num_recommendations=5)

# Get predicted ratings for the recommended lessons from the collaborative model
collab_pred = [test_data[test_data['lesson_id'] == lesson_id]['rating'].values[0] if lesson_id in test_data[
    'lesson_id'].values else 0 for lesson_id in collab_recommendations.index]

# 2. Content-Based Filtering Evaluation
lesson_id = test_data['lesson_id'].values[0]  # Test on the first lesson in the test set
content_recommendations = recommend_content_based(lesson_id, num_recommendations=5)

# Get predicted ratings for the recommended lessons from the content-based model
content_pred = [test_data[test_data['lesson_id'] == lesson_id]['rating'].values[0] if lesson_id in test_data[
    'lesson_id'].values else 0 for lesson_id in content_recommendations]

# 3. Hybrid Model Evaluation
hybrid_recommendations_list = hybrid_recommendations(user_id, num_recommendations=5)

# Get predicted ratings for the recommended lessons from the hybrid model
hybrid_pred = [test_data[test_data['lesson_id'] == lesson_id]['rating'].values[0] if lesson_id in test_data[
    'lesson_id'].values else 0 for lesson_id in hybrid_recommendations_list]

# Evaluate and print performance for each model
evaluate_model(y_true[:len(collab_pred)], collab_pred, "Collaborative Filtering")
evaluate_model(y_true[:len(content_pred)], content_pred, "Content-Based Filtering")
evaluate_model(y_true[:len(hybrid_pred)], hybrid_pred, "Hybrid Model")
