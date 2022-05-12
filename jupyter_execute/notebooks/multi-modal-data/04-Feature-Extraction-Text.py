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


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Movie Synopsis Feature Extraction with Bart text summarization
# 
# In this notebook, will will make use of the BART [model](https://huggingface.co/transformers/model_doc/bart.html) to extract features from movie synopsis. 
# 
# Note: this notebook should be executed from within the below container:
# 
# ```
# docker pull huggingface/transformers-pytorch-gpu
# docker run --gpus=all  --rm -it --net=host -v $PWD:/workspace --ipc=host huggingface/transformers-pytorch-gpu 
# ```
# 
# Then from within the container:
# ```
# cd /workspace
# pip install jupyter jupyterlab
# jupyter server extension disable nbclassic
# jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='admin'
# ```

# First, we install some extra package.

# In[ ]:


get_ipython().system('pip install imdbpy')

# Cuda 11 and A100 support
get_ipython().system('pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html')


# In[4]:


import IPython

IPython.Application.instance().kernel.do_shutdown(True)


# ## Download pretrained BART model
# 
# First, we download a pretrained BART model from HuggingFace library.

# In[1]:


from transformers import BartTokenizer, BartModel
import torch

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartModel.from_pretrained('facebook/bart-large').cuda()


# ## Extracting embeddings for all movie's synopsis
# 
# We will use the average hidden state of the last decoder layer as text feature, comprising 1024 float values.  

# In[2]:


import pickle

with open('movies_info.pkl', 'rb') as f:
    movies_infos = pickle.load(f)['movies_infos']


# In[3]:


import torch
import numpy as np
from tqdm import tqdm

embeddings = {}
for movie, movie_info in tqdm(movies_infos.items()):
    synopsis = None
    synopsis = movie_info.get('synopsis')
    if synopsis is None:
        plots = movie_info.get('plot')
        if plots is not None:
            synopsis = plots[0]
    
    if synopsis is not None:
        inputs = tokenizer(synopsis, return_tensors="pt", truncation=True, max_length=1024).to('cuda')
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        embeddings[movie] = outputs.last_hidden_state.cpu().detach().numpy()


# In[4]:


average_embeddings = {}
for movie in embeddings:
    average_embeddings[movie] = np.mean(embeddings[movie].squeeze(), axis=0)


# In[5]:


with open('movies_synopsis_embeddings-1024.pkl', 'wb') as f:
    pickle.dump({"embeddings": average_embeddings}, f, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




