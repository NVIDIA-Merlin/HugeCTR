#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
# # Creating Multi-Modal Movie Feature Store
# 
# Finally, with both the text and image features ready, we now put the multi-modal movie features into a unified feature store.
# 
# If you have downloaded the real data and proceeded through the feature extraction process in notebooks 03-05, then proceed to create the feature store. Else, skip to the `Synthetic data` section below to create random features.
# 
# ## Real data
# 

# In[5]:


import pickle

with open('movies_poster_features.pkl', 'rb') as f:
    poster_feature = pickle.load(f)["feature_dict"]
    
len(poster_feature)


# In[6]:


with open('movies_synopsis_embeddings-1024.pkl', 'rb') as f:
    text_feature = pickle.load(f)["embeddings"]


# In[7]:


len(text_feature)


# In[8]:


import pandas as pd
links = pd.read_csv("./data/ml-25m/links.csv", dtype={"imdbId": str})


# In[9]:


links.shape


# In[10]:


links.head()


# In[11]:


poster_feature['0105812'].shape


# In[12]:


import numpy as np
feature_array = np.zeros((len(links), 1+2048+1024))

for i, row in links.iterrows():
    feature_array[i,0] = row['movieId']
    if row['imdbId'] in poster_feature:
        feature_array[i,1:2049] = poster_feature[row['imdbId']]
    if row['movieId'] in text_feature:
        feature_array[i,2049:] = text_feature[row['movieId']]
    


# In[13]:


dtype= {**{'movieId': np.int64},**{x: np.float32 for x in ['poster_feature_%d'%i for i in range(2048)]+['text_feature_%d'%i for i in range(1024)]}}


# In[14]:


len(dtype)


# In[15]:


feature_df = pd.DataFrame(feature_array, columns=['movieId']+['poster_feature_%d'%i for i in range(2048)]+['text_feature_%d'%i for i in range(1024)])


# In[16]:


feature_df.head()


# In[17]:


feature_df.shape


# In[18]:


get_ipython().system('pip install pyarrow')


# In[19]:


feature_df.to_parquet('feature_df.parquet')


# 
# ## Synthetic data
# 
# If you have not extrated image and text features from real data, proceed with this section to create synthetic features.

# In[1]:


import pandas as pd
links = pd.read_csv("./data/ml-25m/links.csv", dtype={"imdbId": str})


# In[9]:


import numpy as np

feature_array = np.random.rand(links.shape[0], 3073)


# In[10]:


feature_array[:,0] = links['movieId'].values


# In[11]:


feature_df = pd.DataFrame(feature_array, columns=['movieId']+['poster_feature_%d'%i for i in range(2048)]+['text_feature_%d'%i for i in range(1024)])


# In[12]:


feature_df.to_parquet('feature_df.parquet')


# In[13]:


feature_df.head()


# In[ ]:




