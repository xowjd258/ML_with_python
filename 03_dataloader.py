#!/usr/bin/env python
# coding: utf-8

# In[8]:


# easy_install flickrapi

import requests
import os
import flickrapi
from flickr_keys import *
from PIL import Image


def resizing(fname, dimension, outfname):
    image = Image.open(fname)
    image = image.resize(dimension, Image.ANTIALIAS)  # ex, dimension = (32, 32)
    image.save(outfname)
    os.remove(fname)


def flickr_crawl(keyword, style):
    flickr = flickrapi.FlickrAPI(KEY, SECRET, cache=True)


    photos = flickr.walk(text=keyword,
                         tag_mode='all',
                         tags=keyword,
                         extras=style,
                         per_page=100,  # may be you can try different numbers..
                         sort='relevance')

    urls = []
    for i, photo in enumerate(photos):
        url = photo.get(style)
        urls.append(url)

        # get 50 urls
        if i > 500:
            break
    return urls


def get_path(tag):
    cwd = os.getcwd()
    # generate root folder
    root_path = os.path.join(cwd, 'data')
    img_path = os.path.join(root_path, tag)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    return img_path


def download_img(img_url, fname):
    img_data = requests.get(img_url).content

    with open(fname, 'wb') as handler:
        handler.write(img_data)


keywords = ['cat', 'dog']
style = 'url_sq'
dimension = (32, 32)

for keyword in keywords:
    img_path = get_path(keyword)
    urls = flickr_crawl(keyword, style)

    for idx, url in enumerate(urls):
        fname = keyword + f'{idx:04d}.jpg'
        fname = os.path.join(img_path, fname)
        download_img(url, fname)
        outfname = f'resized_{keyword}_{idx:04d}.jpg'
        outfname = os.path.join(img_path, outfname)
        resizing(fname, dimension, outfname)


# In[5]:


import torchvision #토치비전 써서 데이터를 로드
from torchvision import transforms

from torch.utils.data import DataLoader #데이터로더를 이용해서 트레이닝하는 애한테 떠먹여줌


# In[6]:


trans = transforms.Compose([
    transforms.ToTensor()
])

train_data = torchvision.datasets.ImageFolder(root='data', transform=trans)
train_data_loader = DataLoader(dataset=train_data, batch_size=25, shuffle=True, drop_last=True)


# In[7]:


for X,Y in train_data_loader:
    print(X,Y)
    break


# In[ ]:





# In[ ]:





# In[ ]:




