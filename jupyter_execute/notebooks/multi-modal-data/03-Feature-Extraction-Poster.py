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
# # Movie Poster Feature Extraction with ResNet
# 
# In this notebook, we will use a pretrained ResNet-50 network to extract image features from the movie poster images. 
# 
# Note: this notebook should be executed from within the `nvidia_resnet50` container, built as follows
# ```
# git clone https://github.com/NVIDIA/DeepLearningExamples
# git checkout 5d6d417ff57e8824ef51573e00e5e21307b39697
# cd DeepLearningExamples/PyTorch/Classification/ConvNets
# docker build . -t nvidia_resnet50
# ```
# 
# Start the container, mounting the current directory:
# 
# ```
# nvidia-docker run --rm --net=host -it -v $PWD:/workspace --ipc=host nvidia_resnet50
# ```
# 
# Then from within the container:
# 
# ```
# cd /workspace
# jupyter-lab --allow-root --ip='0.0.0.0'
# 
# ```

# ## Download a pretrained ResNet-50 from NVIDIA GPU cloud
# 
# First, we install an extra package and restart the kernel.

# In[ ]:


get_ipython().system('pip install ipywidgets tqdm')
import IPython

IPython.Application.instance().kernel.do_shutdown(True)


# In[1]:


from PIL import Image
import argparse
import numpy as np
import json
import torch
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn

import sys
sys.path.append('/workspace/DeepLearningExamples/PyTorch/Classification/ConvNets')
from image_classification import models
import torchvision.transforms as transforms


# In[2]:


from image_classification.models import (
    resnet50,
    resnext101_32x4d,
    se_resnext101_32x4d,
    efficientnet_b0,
    efficientnet_b4,
    efficientnet_widese_b0,
    efficientnet_widese_b4,
    efficientnet_quant_b0,
    efficientnet_quant_b4,
)


# In[3]:


def available_models():
    models = {
        m.name: m
        for m in [
            resnet50,
            resnext101_32x4d,
            se_resnext101_32x4d,
            efficientnet_b0,
            efficientnet_b4,
            efficientnet_widese_b0,
            efficientnet_widese_b4,
            efficientnet_quant_b0,
            efficientnet_quant_b4,
        ]
    }
    return models


# In[4]:


def load_jpeg_from_file(path, image_size, cuda=True):
    img_transforms = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    img = img_transforms(Image.open(path))
    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        img = img.float()

        if img.shape[0] == 1: #mono image
            #pad channels
            img = img.repeat([3, 1, 1])
        input = img.unsqueeze(0).sub_(mean).div_(std)

    return input


# In[5]:


def check_quant_weight_correctness(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    quantizers_sd_keys = {f'{n[0]}._amax' for n in model.named_modules() if 'quantizer' in n[0]}
    sd_all_keys = quantizers_sd_keys | set(model.state_dict().keys())
    assert set(state_dict.keys()) == sd_all_keys, (f'Passed quantized architecture, but following keys are missing in '
                                                   f'checkpoint: {list(sd_all_keys - set(state_dict.keys()))}')


# In[6]:


imgnet_classes = np.array(json.load(open("/workspace/DeepLearningExamples/PyTorch/Classification/ConvNets/LOC_synset_mapping.json", "r")))

model_args = {}
model_args["pretrained_from_file"] = './nvidia_resnet50_200821.pth.tar'
model = available_models()['resnet50'](model_args)

model = model.cuda()
model.eval()


# ## Extract features for all movies
# 
# Next, we will extract feature for all movie posters, using the last layer just before the classification head, containing 2048 float values.

# In[7]:


import glob

filelist = glob.glob('./poster_small/*.jpg')
len(filelist)


# In[8]:


filelist[:10]


# In[9]:


from tqdm import tqdm

batchsize = 64
num_bathces = len(filelist)//batchsize
batches = np.array_split(filelist, num_bathces)


# In[10]:


### strip the last layer
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

feature_dict = {}
error = 0
for batch in tqdm(batches):
    inputs = []
    imgs = []
    for i, f in enumerate(batch):
        try:
            img = load_jpeg_from_file(f, 224, cuda=True)
            imgs.append(f.split('/')[-1].split('.')[0])
            inputs.append(img.squeeze())
        except Exception as e:
            print(e)
            error +=1
    features = feature_extractor(torch.stack(inputs, dim=0)).cpu().detach().numpy().squeeze()  
    for i, f in enumerate(imgs):
        feature_dict[f] =features[i,:]

print('Unable to extract features for %d images'%error)


# In[11]:


import pickle
with open('movies_poster_features.pkl', 'wb') as f:
    pickle.dump({"feature_dict": feature_dict}, f, protocol=pickle.HIGHEST_PROTOCOL)


# In[12]:


len(feature_dict)

