import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np


from make_prediction import recommend_collaborative, recommend_content_based, hybrid_recommendations

df_interactions = pd.read_csv("preprocessed_interactions.csv")

train_data, test_data = train_test_split(df_interactions, test_size=0.2, random_state=42)

y_true = test_data['rating'].values

def evaluate_model(y_true, y_pred, model_name):
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > 0).astype(int)

    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)

    precision = precision_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
    mse = mean_squared_error(y_true, y_pred)

    print(f"Performance of {model_name}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"MSE: {mse:.4f}")
    print("-" * 50)


user_id = 1


collab_recommendations = recommend_collaborative(user_id, num_recommendations=5)
collab_pred = [test_data[test_data['lesson_id'] == lesson_id]['rating'].values[0] if lesson_id in test_data[
    'lesson_id'].values else 0 for lesson_id in collab_recommendations.index]


lesson_id = test_data['lesson_id'].values[0]
content_recommendations = recommend_content_based(lesson_id, num_recommendations=5)
content_pred = [test_data[test_data['lesson_id'] == lesson_id]['rating'].values[0] if lesson_id in test_data[
    'lesson_id'].values else 0 for lesson_id in content_recommendations]


hybrid_recommendations_list = hybrid_recommendations(user_id, num_recommendations=5)
hybrid_pred = [test_data[test_data['lesson_id'] == lesson_id]['rating'].values[0] if lesson_id in test_data[
    'lesson_id'].values else 0 for lesson_id in hybrid_recommendations_list]

evaluate_model(y_true[:len(collab_pred)], collab_pred, "Collaborative Filtering")
evaluate_model(y_true[:len(content_pred)], content_pred, "Content-Based Filtering")
evaluate_model(y_true[:len(hybrid_pred)], hybrid_pred, "Hybrid Model")
