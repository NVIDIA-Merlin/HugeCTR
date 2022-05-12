#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
# # MovieLens Data Enrichment
# 
# In this notebook, we will enrich the MovieLens 25M dataset with poster and movie sypnopsis scrapped from IMDB. If you wish to use synthetic multi-modal data, then proceed to [05-Create-Feature-Store.ipynb](05-Create-Feature-Store.ipynb), synthetic data section.
# 
# First, we will need to install some extra package for IMDB data collection.

# In[ ]:


get_ipython().system('pip install imdbpy')


# Note: restart the kernel for the new package to take effect.
# 

# In[ ]:


import IPython

IPython.Application.instance().kernel.do_shutdown(True)


# ## Scraping data from IMDB
# 
# The IMDB API allows the collection of a rich set of multi-modal meta data from the IMDB database, including link to poster, synopsis and plots.

# In[ ]:


from imdb import IMDb

# create an instance of the IMDb class
ia = IMDb()

# get a movie and print its director(s)
the_matrix = ia.get_movie('0114709')
for director in the_matrix['directors']:
    print(director['name'])

# show all information that are currently available for a movie
print(sorted(the_matrix.keys()))

# show all information sets that can be fetched for a movie
print(ia.get_movie_infoset())


# In[ ]:


print(the_matrix.get('plot'))


# In[ ]:


the_matrix.get('synopsis')


# ## Collect synopsis for all movies
# 
# Next, we will collect meta data, including the synopsis, for all movies in the dataset. Note that this process will take a while to complete.

# In[ ]:


from collections import defaultdict
import pandas as pd


# In[ ]:


links = pd.read_csv("./data/ml-25m/links.csv")


# In[ ]:


links.head()


# In[ ]:


links.imdbId.nunique()


# In[ ]:


from tqdm import tqdm
import pickle
from multiprocessing import Process, cpu_count
from multiprocessing.managers import BaseManager, DictProxy


# In[ ]:


movies = list(links['imdbId'])
movies_id = list(links['movieId'])


# In[ ]:


movies_infos = {}
def task(movies, movies_ids, movies_infos):
    for i, (movie, movies_id) in tqdm(enumerate(zip(movies, movies_ids)), total=len(movies)):        
        try:
            movie_info = ia.get_movie(movie)
            movies_infos[movies_id] = movie_info
        except Exception as e:
            print("Movie %d download error: "%movies_id, e)

#task(movies, movies_ids, movies_infos)


# We will now collect the movie metadata from IMDB using parallel threads.
# 
# Please note: with higher thread counts, there is a risk of being blocked by IMDB DoS software.

# In[ ]:


print ('Gathering movies information from IMDB...')
BaseManager.register('dict', dict, DictProxy)
manager = BaseManager()
manager.start()

movies_infos = manager.dict()

num_jobs = 5
total = len(movies)
chunk_size = total // num_jobs + 1
processes = []

for i in range(0, total, chunk_size):
    proc = Process(
        target=task,
        args=[
            movies[i:i+chunk_size],
            movies_id[i:i+chunk_size],
            movies_infos
        ]
    )
    processes.append(proc)
for proc in processes:
    proc.start()
for proc in processes:
    proc.join()


# In[ ]:


movies_infos = movies_infos.copy()


# In[ ]:


len(movies_infos)


# In[ ]:


with open('movies_info.pkl', 'wb') as f:
    pickle.dump({"movies_infos": movies_infos}, f, protocol=pickle.HIGHEST_PROTOCOL)


# ## Scraping movie posters
# 
# The movie metadata also contains link to poster images. We next collect these posters where available.
# 
# Note: this process will take some time to complete.

# In[ ]:


from multiprocessing import Process, cpu_count
import pickle
import subprocess
from tqdm import tqdm
import os

with open('movies_info.pkl', 'rb') as f:
    movies_infos = pickle.load(f)['movies_infos']


# In[ ]:


COLLECT_LARGE_POSTER = False

filelist, targetlist = [], []
largefilelist, largetargetlist = [], []

for key, movie in tqdm(movies_infos.items(), total=len(movies_infos)):
    if 'cover url' in movie.keys():
        target_path = './poster_small/%s.jpg'%(movie['imdbID'])
        if os.path.exists(target_path):
            continue
        targetlist.append(target_path)
        filelist.append(movie['cover url'])
                
    # Optionally, collect high-res poster images 
    if COLLECT_LARGE_POSTER:
        if 'full-size cover url' in movie.keys():
            target_path = '"./poster_large/%s.jpg"'%(movie['imdbID'])
            if os.path.exists(target_path):
                continue
            largetargetlist.append(target_path)
            largefilelist.append(movie['full-size cover url'])                                


# In[ ]:


def download_task(filelist, targetlist):
    for i, (file, target) in tqdm(enumerate(zip(filelist, targetlist)), total=len(targetlist)):        
        cmd = 'wget "%s" -O %s'%(file, target)
        stream = os.popen(cmd)
        output = stream.read()
        print(output, cmd)


# In[ ]:


print ('Gathering small posters...')
get_ipython().system('mkdir ./poster_small')

num_jobs = 10
total = len(filelist)
chunk_size = total // num_jobs + 1
processes = []

for i in range(0, total, chunk_size):
    proc = Process(
        target=download_task,
        args=[
            filelist[i:i+chunk_size],
            targetlist[i:i+chunk_size],            
        ]
    )
    processes.append(proc)
for proc in processes:
    proc.start()
for proc in processes:
    proc.join()


# In[ ]:


if COLLECT_LARGE_POSTER:
    print ('Gathering large posters...')
    get_ipython().system('mkdir ./poster_large')

    num_jobs = 32
    total = len(largefilelist)
    chunk_size = total // num_jobs + 1
    processes = []

    for i in range(0, total, chunk_size):
        proc = Process(
            target=download_task,
            args=[
                largefilelist[i:i+chunk_size],
                largetargetlist[i:i+chunk_size],            
            ]
        )
        processes.append(proc)
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()


# In[ ]:


get_ipython().system('ls -l poster_small|wc -l')


# In[ ]:




