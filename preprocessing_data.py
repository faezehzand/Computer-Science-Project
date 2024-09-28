import pandas as pd

df_interactions = pd.read_csv("simulated_user_interactions.csv")
df_lessons = pd.read_csv("lesson_attributes.csv")
df_interactions.drop_duplicates(subset=['user_id', 'lesson_id'], inplace=True)

print(f"Missing values in interactions: \n{df_interactions.isnull().sum()}")
print(f"Missing values in lessons: \n{df_lessons.isnull().sum()}")

df_interactions['rating'] = df_interactions['rating'].fillna(0)
df_interactions['time_spent'] = df_interactions['time_spent'].fillna(df_interactions['time_spent'].mean())
df_interactions['quiz_score'] = df_interactions['quiz_score'].fillna(df_interactions['quiz_score'].mean())

df_interactions['timestamp'] = pd.to_datetime(df_interactions['timestamp'])

df_interactions['time_spent_norm'] = (df_interactions['time_spent'] - df_interactions['time_spent'].mean()) / df_interactions['time_spent'].std()
df_interactions['quiz_score_norm'] = (df_interactions['quiz_score'] - df_interactions['quiz_score'].mean()) / df_interactions['quiz_score'].std()

df = pd.merge(df_interactions, df_lessons, on='lesson_id')

user_item_matrix = df.pivot_table(index='user_id', columns='lesson_id', values='rating').fillna(0)

user_item_matrix_normalized = user_item_matrix.sub(user_item_matrix.mean(axis=1), axis=0)

df.to_csv("preprocessed_interactions.csv", index=False)
user_item_matrix.to_csv("user_item_matrix.csv", index=True)

print(df.head())
