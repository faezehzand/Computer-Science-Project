import pandas as pd

# Load the simulated user interactions and lesson attributes datasets
df_interactions = pd.read_csv("simulated_user_interactions.csv")
df_lessons = pd.read_csv("lesson_attributes.csv")

# Remove duplicate entries where the same user interacts with the same lesson more than once
df_interactions.drop_duplicates(subset=['user_id', 'lesson_id'], inplace=True)

# Print missing value counts for both datasets to identify any gaps in the data
print(f"Missing values in interactions: \n{df_interactions.isnull().sum()}")
print(f"Missing values in lessons: \n{df_lessons.isnull().sum()}")

# Handle missing values by filling them with appropriate values
df_interactions['rating'] = df_interactions['rating'].fillna(0)  # Replace missing ratings with 0
df_interactions['time_spent'] = df_interactions['time_spent'].fillna(df_interactions['time_spent'].mean())  # Fill missing time spent with mean
df_interactions['quiz_score'] = df_interactions['quiz_score'].fillna(df_interactions['quiz_score'].mean())  # Fill missing quiz scores with mean

# Convert the timestamp column to a datetime format for consistency
df_interactions['timestamp'] = pd.to_datetime(df_interactions['timestamp'])

# Normalize the 'time_spent' and 'quiz_score' columns to bring them into a standard range
df_interactions['time_spent_norm'] = (df_interactions['time_spent'] - df_interactions['time_spent'].mean()) / df_interactions['time_spent'].std()
df_interactions['quiz_score_norm'] = (df_interactions['quiz_score'] - df_interactions['quiz_score'].mean()) / df_interactions['quiz_score'].std()

# Merge the user interaction data with lesson attributes to form a complete dataset
df = pd.merge(df_interactions, df_lessons, on='lesson_id')

# Create a user-item matrix (rows represent users, columns represent lessons) with ratings as values
user_item_matrix = df.pivot_table(index='user_id', columns='lesson_id', values='rating').fillna(0)

# Normalize the user-item matrix by subtracting the mean rating for each user (user-based normalization)
user_item_matrix_normalized = user_item_matrix.sub(user_item_matrix.mean(axis=1), axis=0)

# Save the preprocessed datasets to CSV files
df.to_csv("preprocessed_interactions.csv", index=False)  # Save the merged and preprocessed interaction data
user_item_matrix.to_csv("user_item_matrix.csv", index=True)  # Save the user-item matrix

# Display the first few rows of the merged dataset for verification
print(df.head())
