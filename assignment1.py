import numpy as np
import time
import re #regular expressions

from pandas import Series
import pandas as pd
from pandas import DataFrame

import matplotlib
import matplotlib.pyplot as plt


u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.user',
    sep='|', names=u_cols
)

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.data',
    sep='\t', names=r_cols
)

m_cols = ['movie_id', 'title', 'release_date',
          'video_release_date', 'imdb_url']
movies = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.item',
                     sep='|', names=m_cols, usecols=range(5), encoding='latin-1')

# Question 1
# show users aged 40 and Male
aged_40_male = users[(users.age == 40) & (users.sex == 'M')]
#print(aged_40_male)

# Question 2
# show users who are Female and programmers
female_programmers = users[(users.sex == 'F') & (users.occupation == 'programmer')]
#print(female_programmers)

# Question 3
# Compute mean of ratings
ratings_mean = ratings['rating'].mean()
#print(ratings_mean)

# Question 4
# What is the average rating of female users?
user_ratings = pd.concat([users, ratings], axis=1, join='inner')  # join tables: users and ratings
females = user_ratings[(user_ratings.sex == 'F')]  # get female users from joint table
print(females)
female_mean = females['rating'].mean()  # calculate mean of ratings
#print(female_mean)  # result = 3.5604395604395602

# What is the average rating of male users?
males = user_ratings[(user_ratings.sex == 'M')]
male_mean = males['rating'].mean()
#print(male_mean)  # result = 3.508955223880597

# Question 5
# What is the number of female users who rated Toy Story (1995)?
toy_story_id = int(movies[(movies.title == 'Toy Story (1995)')].movie_id) # to get the movie_id primary key of movie
print(toy_story_id)  # movie_id of Toy Story (1995) = 1
rated_females = user_ratings[(user_ratings.sex == 'F') & (user_ratings.movie_id == 1)]
print(rated_females.shape[0])
