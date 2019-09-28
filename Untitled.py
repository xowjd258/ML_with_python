#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import time

from numba import vectorize, cuda


# In[4]:


def vectorAdd(a,b):
    return a+b


# In[6]:


def main():
    N= 3200000
    A=np.ones(N, dtype=np.float32)
    B=np.ones(N, dtype=np.float32)
    start = time.time()
    C= vectorAdd(A,B)
    vector_add_time = time.time() -start
    print("C[:5] = "+str(C[:5]))
    print("C[-5:]= "+str(C[-5:]))
    print("VectorAdd took for % seconds"% vector_add_time)
if_name_=='_main_':
    main()


# In[ ]:





# In[ ]:




