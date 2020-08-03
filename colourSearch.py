#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#authour: aaron edwards
#[ colours ] search engine using [ tf/idf ]
#02 aug 2020

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


colours = pd.read_csv("colours.csv", sep=',', header=None)
colours.head()


# In[ ]:


#drop extra row 
colours = colours.drop([0], axis=1) 
colours.head()


# In[ ]:


#create tf / idf object
tvec = TfidfVectorizer()


# In[ ]:


colours.rename(columns = {colours.columns[0]:'ColourNames', 
                          colours.columns[1]: 'HexColour',
                          colours.columns[2]: 'HueValue1',
                          colours.columns[3]: 'HueValue2', 
                          colours.columns[4]: 'HueValue3'
                         }, inplace = True)
colours.head()


# In[ ]:


#use the fit_transform method to convert words to tf / idf score ; save as X
needed = colours[['ColourNames', "HexColour"]]
        


# In[ ]:


needed.head()


# In[ ]:


X = tvec.fit_transform(colours['ColourNames'])


# In[ ]:


print("Please Enter Colour below...")
query = input()


# In[ ]:


#get the tf / idf vectoriser representation of the query using the transform method 
query_vec = tvec.transform([query])


# In[ ]:


#oT-List
oT = cosine_similarity(X, query_vec).reshape((-1,))


# In[ ]:


for i in oT.argsort()[10:][::-1]:
    print(needed.iloc[i, 0], " -- ", needed.iloc[i, 1])


# In[ ]:




