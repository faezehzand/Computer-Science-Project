import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("preprocessed_interactions.csv")

# Set plot style for consistency
sns.set(style="whitegrid")

# basic dataset overview

# print the first few rows of the dataset to verify structure
print(df.head())

# summary statistics for the dataset
print(df.describe())

print(f"Rating distribution: \n{df['rating'].value_counts()}")

print(f"Number of unique users: {df['user_id'].nunique()}")
print(f"Number of unique lessons: {df['lesson_id'].nunique()}")

print(f"Missing values in the dataset: \n{df.isnull().sum()}")

# EDA - Visualizations

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='rating', palette='Blues')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

popular_lessons = df['lesson_id'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=popular_lessons.index, y=popular_lessons.values, palette='Purples')
plt.title('Top 10 Most Popular Lessons')
plt.xlabel('Lesson ID')
plt.ylabel('Number of Interactions')
plt.show()

avg_rating_per_lesson = df.groupby('lesson_id')['rating'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_rating_per_lesson.index, y=avg_rating_per_lesson.values, palette='Greens')
plt.title('Top 10 Lessons with Highest Average Rating')
plt.xlabel('Lesson ID')
plt.ylabel('Average Rating')
plt.show()

user_interactions = df['user_id'].value_counts()
plt.figure(figsize=(10, 6))
sns.histplot(user_interactions, bins=20, color='orange')
plt.title('Distribution of User Interactions')
plt.xlabel('Number of Interactions per User')
plt.ylabel('Number of Users')
plt.show()


avg_time_spent = df.groupby('lesson_id')['time_spent'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_time_spent.index, y=avg_time_spent.values, palette='Reds')
plt.title('Top 10 Lessons with Longest Average Time Spent')
plt.xlabel('Lesson ID')
plt.ylabel('Average Time Spent (minutes)')
plt.show()


completion_rates = df.groupby('activity_type')['activity_type'].count()
plt.figure(figsize=(8, 6))
sns.barplot(x=completion_rates.index, y=completion_rates.values, palette='coolwarm')
plt.title('Views vs. Completions')
plt.xlabel('Activity Type')
plt.ylabel('Count')
plt.show()

# correlation analysis

plt.figure(figsize=(10, 8))
corr_matrix = df[['rating', 'time_spent', 'quiz_score']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Numeric Features')
plt.show()

# advanced insights

# users who viewed vs. completed the most lessons
user_activity_counts = df.groupby(['user_id', 'activity_type']).size().unstack(fill_value=0)
user_activity_counts['total_interactions'] = user_activity_counts.sum(axis=1)
most_active_users = user_activity_counts.sort_values(by='total_interactions', ascending=False).head(10)

print(f"Top 10 Most Active Users:\n{most_active_users}")

# lessons with the highest quiz scores
top_quiz_scores = df.groupby('lesson_id')['quiz_score'].mean().sort_values(ascending=False).head(10)
print(f"Top 10 Lessons with Highest Quiz Scores:\n{top_quiz_scores}")

