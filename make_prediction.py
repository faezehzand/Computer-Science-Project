import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Load the preprocessed interaction data and lesson attributes
df_interactions = pd.read_csv("preprocessed_interactions.csv")
df_lessons = pd.read_csv("lesson_attributes.csv")

# Create a user-item matrix where rows are users, columns are lessons, and values are ratings
user_item_matrix = df_interactions.pivot_table(index='user_id', columns='lesson_id', values='rating').fillna(0)

# Normalize the user-item matrix by subtracting the mean rating for each user
user_item_matrix_normalized = user_item_matrix.sub(user_item_matrix.mean(axis=1), axis=0)

# Compute the cosine similarity between users based on their ratings
user_similarity = cosine_similarity(user_item_matrix_normalized)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


def recommend_collaborative(user_id, num_recommendations=5):
    """
    Recommend lessons using collaborative filtering for a specific user.

    Parameters:
    user_id (int): The ID of the user to recommend lessons to.
    num_recommendations (int): Number of recommendations to provide.

    Returns:
    pd.Series: A series of recommended lessons with predicted ratings.
    """
    # Get similarity scores for the given user
    sim_scores = user_similarity_df[user_id]

    # Find users most similar to the target user, excluding the user themselves
    similar_users = sim_scores.sort_values(ascending=False).index[1:]

    # Find lessons the user has not rated
    lessons_rated_by_user = user_item_matrix.loc[user_id]
    unrated_lessons = lessons_rated_by_user[lessons_rated_by_user == 0].index

    # Aggregate lesson ratings from similar users
    lesson_recommendations = pd.Series(dtype=float)
    for similar_user in similar_users:
        similar_user_ratings = user_item_matrix.loc[similar_user]
        recommendations = similar_user_ratings[unrated_lessons]
        lesson_recommendations = lesson_recommendations.add(recommendations, fill_value=0)

        if len(lesson_recommendations) >= num_recommendations:
            break

    # Return the top lesson recommendations
    return lesson_recommendations.sort_values(ascending=False).head(num_recommendations)


# Create a lesson-topic matrix using one-hot encoding for the lesson topics
lesson_topic_matrix = pd.get_dummies(df_lessons.set_index('lesson_id')['topic'])

# Compute the cosine similarity between lessons based on their topics
lesson_similarity = cosine_similarity(lesson_topic_matrix)
lesson_similarity_df = pd.DataFrame(lesson_similarity, index=lesson_topic_matrix.index,
                                    columns=lesson_topic_matrix.index)


def recommend_content_based(lesson_id, num_recommendations=5):
    """
    Recommend similar lessons using content-based filtering based on lesson topics.

    Parameters:
    lesson_id (int): The ID of the lesson to find similar lessons to.
    num_recommendations (int): Number of recommendations to provide.

    Returns:
    pd.Index: A list of similar lesson IDs.
    """
    # Get similarity scores for the given lesson
    similar_lessons = lesson_similarity_df[lesson_id]
    # Return the most similar lessons, excluding the lesson itself
    return similar_lessons.sort_values(ascending=False).index[1:num_recommendations + 1]


def hybrid_recommendations(user_id, num_recommendations=5):
    """
    Provide hybrid recommendations using a combination of collaborative filtering and content-based filtering.

    Parameters:
    user_id (int): The ID of the user to recommend lessons to.
    num_recommendations (int): Number of recommendations to provide.

    Returns:
    pd.Series: A series of hybrid lesson recommendations.
    """
    # Get collaborative filtering recommendations
    collab_recommendations = recommend_collaborative(user_id, num_recommendations)

    # Sort recommendations by predicted ratings
    collab_recommendations_sorted = collab_recommendations.sort_values(ascending=False)

    # Initialize list for hybrid recommendations
    hybrid_recommendations = []

    # For each collaborative recommendation, find similar lessons using content-based filtering
    for lesson_id in collab_recommendations_sorted.index:
        similar_lessons = recommend_content_based(lesson_id, num_recommendations)

        if not similar_lessons.empty:
            hybrid_recommendations.extend(similar_lessons.tolist())

    # Return unique hybrid recommendations (removing duplicates)
    hybrid_recommendations_series = pd.Series(hybrid_recommendations)
    return hybrid_recommendations_series.drop_duplicates().head(num_recommendations)


# Test the recommendation system

user_id = 1
print(f"Collaborative Recommendations for User {user_id}:")
print(recommend_collaborative(user_id, num_recommendations=5))

lesson_id = 101
print(f"\nContent-Based Recommendations for Lesson {lesson_id}:")
print(recommend_content_based(lesson_id, num_recommendations=5))

print(f"\nHybrid Recommendations for User {user_id}:")
print(hybrid_recommendations(user_id, num_recommendations=5))
