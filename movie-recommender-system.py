#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
# pd.set_option('display.max_columns',30)
# print(movies.head(1))
# pd.set_option('display.max_columns',30)
# credits.head(1)['cast'].values


# In[3]:


# movies.shape


# In[4]:


# credits.head()


# In[5]:


movies = movies.merge(credits, on="title")


# In[6]:


# movies.head()


# In[7]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[8]:


# movies.head()


# In[9]:


# movies.info()


# In[10]:


# movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


# movies.duplicated().sum()


# In[13]:


# movies.iloc[0].genres


# In[14]:


import ast
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[15]:


movies['genres'] = movies['genres'].apply(convert)


# In[16]:


# movies.head()


# In[17]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[18]:


# movies.head()


# In[19]:


# movies['genres'] = movies['genres'].apply(convert)


# In[20]:


import ast
def convert2(obj):
    counter = 0
    L = []
    for i in ast.literal_eval(obj):
        if (counter!=3):
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[21]:


movies['cast'] = movies['cast'].apply(convert2)


# In[22]:


# movies.head()


# In[23]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[24]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[25]:


# movies.head()


# In[26]:


movies['summary'] = movies['overview']
movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[27]:


movies['overview'][0]


# In[28]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[29]:


movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[30]:


movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[31]:


movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[32]:


# movies.head()


# In[33]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[34]:


# movies.head()


# In[35]:


new_df = movies[['movie_id', 'title', 'tags', 'summary']]


# In[36]:


# new_df.head()


# In[37]:


new_df['tags'][0]
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[38]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[39]:


# new_df.head()


# In[40]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[41]:


# new_df['tags'][1]
# used for stemming of the list 
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
# used to convert into string
    return " ".join(y)
new_df['tags'] = new_df['tags'].apply(stem)


# In[42]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000 ,stop_words = 'english')


# In[43]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[44]:


# vectors


# In[45]:


# cv.get_feature_names()


# In[46]:


from sklearn.metrics.pairwise import cosine_similarity


# In[47]:


# We are storing all the calculated distances in a variable called similarity
similarity = cosine_similarity(vectors)


# In[48]:


# This should take one movie as input, and return 5 movies similar to it
def recommend(movie):
#     This will give the index position
    movie_index = new_df[new_df['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x:x[1])[1:6]
    for i in movies_list:
#         print (new_df['title'].new_df['index'] == i[0])
        print (new_df.iloc[i[0]].title)


# In[49]:


# This will give the index position
# new_df[new_df['title'] == "Avatar"].index[0]


# In[50]:
# We have to put our input here

recommend('Spectre')


# In[ ]:




