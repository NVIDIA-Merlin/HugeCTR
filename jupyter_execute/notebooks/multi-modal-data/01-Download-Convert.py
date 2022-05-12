#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Copyright 2021 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # MovieLens-25M: Download and Convert
# 
# The [MovieLens-25M](https://grouplens.org/datasets/movielens/25m/) is a popular dataset in the recommender systems domain, containing 25M movie ratings for ~62,000 movies given by ~162,000 users. 
# 
# In this notebook, we will download and convert this dataset to a suitable format for subsequent processing.

# ## Getting Started

# In[12]:


# External dependencies
import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from nvtabular.utils import download_file


# We define our base input directory, containing the data.

# In[13]:


INPUT_DATA_DIR = "./data"


# We will download and unzip the data.

# In[21]:


from os.path import exists

if not  exists(os.path.join(INPUT_DATA_DIR, "ml-25m.zip")):
    download_file("http://files.grouplens.org/datasets/movielens/ml-25m.zip", 
              os.path.join(INPUT_DATA_DIR, "ml-25m.zip"))


# ## Convert the dataset

# First, we take a look on the movie metadata. 

# In[15]:


movies = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'ml-25m/movies.csv'))
movies.head()


# We can see, that genres are a multi-hot categorical features with different number of genres per movie. Currently, genres is a String and we want split the String into a list of Strings. In addition, we drop the title.

# In[16]:


movies = movies.drop(['title', 'genres'], axis=1)
movies.head()


# We save movies genres in parquet format, so that they can be used by NVTabular in the next notebook.

# In[17]:


movies.to_parquet(os.path.join(INPUT_DATA_DIR, "movies_converted.parquet"))


# ## Splitting into train and validation dataset

# We load the movie ratings.

# In[18]:


ratings = pd.read_csv(os.path.join(INPUT_DATA_DIR, "ml-25m", "ratings.csv"))
ratings.head()


# We drop the timestamp column and split the ratings into training and test dataset. We use a simple random split.

# In[19]:


ratings = ratings.drop('timestamp', axis=1)
train, valid = train_test_split(ratings, test_size=0.2, random_state=42)


# We save the dataset to disk.

# In[20]:


train.to_parquet(os.path.join(INPUT_DATA_DIR, "train.parquet"))
valid.to_parquet(os.path.join(INPUT_DATA_DIR, "valid.parquet"))


# ## Next steps
# 
# If you wish to download the real enriched data for the movielens-25m dataset, including movie poster and movie synopsis, then proceed through notebooks 02-04. 
# 
# If you wish to use synthetic multi-modal data, then proceed to notebook [05-Create-Feature-Store.ipynb](05-Create-Feature-Store.ipynb), synthetic data section.

# In[ ]:




