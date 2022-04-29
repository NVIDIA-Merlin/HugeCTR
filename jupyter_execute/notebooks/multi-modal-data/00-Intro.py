#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# # Training Recommender Systems on Multi-modal Data

# ## Overview
# 
# Recommender systems are often trained on tabular data, containing numeric fields (such as item price, numbers of user's purchases) and categorical fields (such as user and item IDs).
# 
# Multi-modal data refer to data types in other modalities, such as text, image and video. Such data can additionally provide rich inputs to and potentially improve the effectiveness of recommender systems.
# 
# Several examples include:
# - Movie recommendation, where movie poster, plot and synopsis can be used.
# - Music recommendation, where audio features and lyric can be used.
# - Itinerary planning and attractions recommendation, where text (user profile, attraction description & review) and photos can be used.
# 
# Often times, features from multi-modal data are extracted using domain-specific networks, such as ResNet for images and BERT for text data. These pretrained features, also called pretrained embeddings, are then combined with other trainable features and embeddings for the task of recommendation.

# This series of notebooks demonstrate the use of multi-modal data (text, image) for the task of movie recommendation, using the Movielens-25M dataset.
# 
# - [01-Download-Convert.ipynb](01-Download-Convert.ipynb): download and convert the raw data
# - [02-Data-Enrichment.ipynb](02-Data-Enrichment.ipynb): enrich the tabular data with image and text data 
# - [03-Feature-Extraction-Poster.ipynb](03-Feature-Extraction-Poster.ipynb): extract image features from movie posters
# - [04-Feature-Extraction-Text.ipynb](04-Feature-Extraction-Text.ipynb): extract text features from movie synopsis
# - [05-Create-Feature-Store.ipynb](05-Create-Feature-Store.ipynb): create a combined feature store
# - [06-ETL-with-NVTabular.ipynb](06-ETL-with-NVTabular.ipynb): feature transform with NVTabular
# - [07-Training-with-HugeCTR.ipynb](07-Training-with-HugeCTR.ipynb): train model with HugeCTR, making use of pretrained embeddings.

# In[ ]:




