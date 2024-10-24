import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

df_interactions = pd.read_csv("preprocessed_interactions.csv")
df_lessons = pd.read_csv("lesson_attributes.csv")




user_item_matrix = df_interactions.pivot_table(index='user_id', columns='lesson_id', values='rating').fillna(0)


user_item_matrix_normalized = user_item_matrix.sub(user_item_matrix.mean(axis=1), axis=0)


user_similarity = cosine_similarity(user_item_matrix_normalized)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)



def recommend_collaborative(user_id, num_recommendations=5):
    sim_scores = user_similarity_df[user_id]

    similar_users = sim_scores.sort_values(ascending=False).index[1:]

    lessons_rated_by_user = user_item_matrix.loc[user_id]
    unrated_lessons = lessons_rated_by_user[lessons_rated_by_user == 0].index

    lesson_recommendations = pd.Series(dtype=float)
    for similar_user in similar_users:
        similar_user_ratings = user_item_matrix.loc[similar_user]
        recommendations = similar_user_ratings[unrated_lessons]
        lesson_recommendations = lesson_recommendations.add(recommendations, fill_value=0)

        if len(lesson_recommendations) >= num_recommendations:
            break

    return lesson_recommendations.sort_values(ascending=False).head(num_recommendations)

lesson_topic_matrix = pd.get_dummies(df_lessons.set_index('lesson_id')['topic'])

lesson_similarity = cosine_similarity(lesson_topic_matrix)
lesson_similarity_df = pd.DataFrame(lesson_similarity, index=lesson_topic_matrix.index,
                                    columns=lesson_topic_matrix.index)

def recommend_content_based(lesson_id, num_recommendations=5):
    similar_lessons = lesson_similarity_df[lesson_id]
    return similar_lessons.sort_values(ascending=False).index[1:num_recommendations + 1]  # Exclude the lesson itself

def hybrid_recommendations(user_id, num_recommendations=5):
    collab_recommendations = recommend_collaborative(user_id, num_recommendations)

    collab_recommendations_sorted = collab_recommendations.sort_values(ascending=False)

    hybrid_recommendations = []

    for lesson_id in collab_recommendations_sorted.index:
        similar_lessons = recommend_content_based(lesson_id, num_recommendations)

        if not similar_lessons.empty:
            hybrid_recommendations.extend(similar_lessons.tolist())

    hybrid_recommendations_series = pd.Series(hybrid_recommendations)
    return hybrid_recommendations_series.drop_duplicates().head(num_recommendations)



user_id = 1
print(f"Collaborative Recommendations for User {user_id}:")
print(recommend_collaborative(user_id, num_recommendations=5))

lesson_id = 101
print(f"\nContent-Based Recommendations for Lesson {lesson_id}:")
print(recommend_content_based(lesson_id, num_recommendations=5))

print(f"\nHybrid Recommendations for User {user_id}:")
print(hybrid_recommendations(user_id, num_recommendations=5))


