import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

# Set parameters for generating the dataset
num_users = 100  # Number of users to simulate
num_lessons = 20  # Number of lessons to simulate
num_interactions = 1000  # Number of interactions to simulate

# Create lists of users and lessons
users = list(range(1, num_users + 1))  # User IDs from 1 to num_users
lessons = list(range(101, 101 + num_lessons))  # Lesson IDs from 101 to 101 + num_lessons

# Define possible lesson attributes (difficulty, topics, lengths)
lesson_difficulties = ['easy', 'medium', 'hard']  # Difficulty levels of lessons
lesson_topics = ['Math', 'Science', 'History', 'Literature']  # Topics of lessons
lesson_lengths = [15, 30, 45, 60]  # Possible lesson durations (in minutes)

# Define possible activity types
activities = ['viewed', 'completed']  # Possible activities a user can perform

# Initialize list for storing lesson attributes
lesson_attributes = []

# Generate lesson attributes dataset
for lesson_id in lessons:
    """
    Generate random attributes (topic, difficulty, length) for each lesson and store them in lesson_attributes.
    """
    topic = random.choice(lesson_topics)
    difficulty = random.choice(lesson_difficulties)
    length = random.choice(lesson_lengths)
    lesson_attributes.append([lesson_id, topic, difficulty, length])

# Convert the lesson attributes into a DataFrame
df_lessons = pd.DataFrame(lesson_attributes, columns=['lesson_id', 'topic', 'difficulty', 'length'])

# Initialize list for storing user interaction data
data = []

# Generate user interaction dataset
for _ in range(num_interactions):
    """
    Simulate random user interactions with lessons.

    For each interaction, randomly select a user, a lesson, an activity (viewed/completed),
    and generate random values for rating, timestamp, time spent on the lesson, and quiz score.
    """
    user_id = random.choice(users)
    lesson_id = random.choice(lessons)
    activity = random.choice(activities)
    rating = random.randint(1, 5)  # Random rating between 1 and 5
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30))  # Random timestamp within the last 30 days

    # Additional attributes for interaction
    time_spent = random.randint(1, 60)  # Time spent on the lesson (in minutes)
    quiz_score = random.randint(0, 100)  # Quiz score from 0 to 100

    # Append the simulated interaction data to the list
    data.append([user_id, lesson_id, activity, rating, timestamp, time_spent, quiz_score])

# Convert the interaction data into a DataFrame
df_interactions = pd.DataFrame(data,
                               columns=['user_id', 'lesson_id', 'activity_type', 'rating', 'timestamp', 'time_spent',
                                        'quiz_score'])

# Save the generated datasets to CSV files
df_interactions.to_csv("simulated_user_interactions.csv", index=False)
df_lessons.to_csv("lesson_attributes.csv", index=False)

# Display the first few rows of both datasets for verification
print(df_interactions.head())
print(df_lessons.head())
