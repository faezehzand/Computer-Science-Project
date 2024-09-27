import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta


num_users = 100
num_lessons = 20
num_interactions = 1000


users = list(range(1, num_users + 1))
lessons = list(range(101, 101 + num_lessons))


lesson_difficulties = ['easy', 'medium', 'hard']
lesson_topics = ['Math', 'Science', 'History', 'Literature']
lesson_lengths = [15, 30, 45, 60]


activities = ['viewed', 'completed']

lesson_attributes = []
for lesson_id in lessons:
    topic = random.choice(lesson_topics)
    difficulty = random.choice(lesson_difficulties)
    length = random.choice(lesson_lengths)
    lesson_attributes.append([lesson_id, topic, difficulty, length])


df_lessons = pd.DataFrame(lesson_attributes, columns=['lesson_id', 'topic', 'difficulty', 'length'])


data = []
for _ in range(num_interactions):
    user_id = random.choice(users)
    lesson_id = random.choice(lessons)
    activity = random.choice(activities)
    rating = random.randint(1, 5)
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30))


    time_spent = random.randint(1, 60)
    quiz_score = random.randint(0, 100)

    data.append([user_id, lesson_id, activity, rating, timestamp, time_spent, quiz_score])


df_interactions = pd.DataFrame(data,
                               columns=['user_id', 'lesson_id', 'activity_type', 'rating', 'timestamp', 'time_spent',
                                        'quiz_score'])


df_interactions.to_csv("simulated_user_interactions.csv", index=False)
df_lessons.to_csv("lesson_attributes.csv", index=False)


print(df_interactions.head())
print(df_lessons.head())
