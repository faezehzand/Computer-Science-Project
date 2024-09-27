import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

# Parameters for the simulation
num_users = 100  # Number of users
num_lessons = 20  # Number of lessons
num_interactions = 1000  # Total number of interactions

# Create lists of user IDs and lesson IDs
users = list(range(1, num_users + 1))  # User IDs from 1 to 100
lessons = list(range(101, 101 + num_lessons))  # Lesson IDs from 101 to 120

# Lesson Attributes for Content-Based Filtering
lesson_difficulties = ['easy', 'medium', 'hard']
lesson_topics = ['Math', 'Science', 'History', 'Literature']
lesson_lengths = [15, 30, 45, 60]  # Lesson durations in minutes

# Activity types: users can view or complete lessons
activities = ['viewed', 'completed']

# Generating additional lesson attributes
lesson_attributes = []
for lesson_id in lessons:
    topic = random.choice(lesson_topics)
    difficulty = random.choice(lesson_difficulties)
    length = random.choice(lesson_lengths)
    lesson_attributes.append([lesson_id, topic, difficulty, length])

# Create a DataFrame for lesson attributes
df_lessons = pd.DataFrame(lesson_attributes, columns=['lesson_id', 'topic', 'difficulty', 'length'])

# Generate random interactions
data = []
for _ in range(num_interactions):
    user_id = random.choice(users)
    lesson_id = random.choice(lessons)
    activity = random.choice(activities)
    rating = random.randint(1, 5)  # Random rating between 1 and 5
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30))  # Random date in the past 30 days

    # Additional user feedback data for hybrid recommendations
    time_spent = random.randint(1, 60)  # Time spent on a lesson (in minutes)
    quiz_score = random.randint(0, 100)  # Quiz score after the lesson

    data.append([user_id, lesson_id, activity, rating, timestamp, time_spent, quiz_score])

# Create a DataFrame from the simulated data
df_interactions = pd.DataFrame(data,
                               columns=['user_id', 'lesson_id', 'activity_type', 'rating', 'timestamp', 'time_spent',
                                        'quiz_score'])

# Save the dataset to CSV files (optional)
df_interactions.to_csv("simulated_user_interactions.csv", index=False)
df_lessons.to_csv("lesson_attributes.csv", index=False)

# Preview the dataset
print(df_interactions.head())
print(df_lessons.head())
