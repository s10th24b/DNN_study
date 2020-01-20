#!/usr/bin/env python
# coding: utf-8

# In[24]:


import tensorflow as tf
print(tf.__version__)
import numpy as np
get_ipython().system('pip3 install --user tensorflow-gpu==2.0.0-alpha0')


# In[ ]:





# In[26]:


# bugs music 에서 다운로드한 11,000여 곡의 한국 힙합 노래 가사 데이터를 불러옴
path_to_file = tf.keras.utils.get_file('input.txt', 'https://raw.githubusercontent.com/greentec/greentec.github.io/master/public/other/data/koreanhiphop/input.txt')
path_to_file


# In[29]:


text = open(path_to_file,'rb').read().decode(encoding='utf-8')
text = text[:len(text)//5]
print("Length of text: {} characters".format(len(text)))


# In[30]:


print(text[:500])


# In[ ]:


import re
def clean_str(string):
    string = re.sub(r")

