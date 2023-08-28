#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


df=pd.read_csv('Restaurant_Reviews (1).tsv',delimiter='\t',quoting=3) 
df  


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


# cleaning the text   pre processing
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')  
stopwords=stopwords.words('english') 
stopwords.remove('not') 
from nltk.stem.porter import PorterStemmer

corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',df['Review'][i])   

    review=review.lower()
    review=review.split()
    ps=PorterStemmer() 
    review=[ps.stem(x) for x in review if x in stopwords]
    review=' '.join(review) 
    corpus.append(review)
    


# In[6]:


corpus


# In[7]:


# create bag of words:

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500) 
x=cv.fit_transform(corpus).toarray() 
x 




# In[8]:


y = df.iloc[:, -1]
y


# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[11]:


# Naive Bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)


# In[12]:


y_pred = classifier.predict(x_test)


# In[13]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[14]:


# svm
from sklearn.svm import SVC   
classifier=SVC(C=5000,kernel="poly") 
classifier.fit(x_train,y_train)


# In[15]:


y_pred=classifier.predict(x_test)
y_pred


# In[16]:


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

# from sklearn.metrics import r2_score
# print(abs(r2_score(y_test,y_pred)))


# In[17]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[18]:


# decision Tree
from sklearn.tree import DecisionTreeClassifier  # use the decision tree classifier 
classifier=DecisionTreeClassifier(max_depth=10,min_samples_split=20)    # max_depth means 10 kai bad training stop karde taki overfiting na ho
classifier.fit(x_train,y_train)


# In[19]:


y_pred=classifier.predict(x_test)


# In[20]:


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))


# In[21]:


# logistic
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)


# In[22]:


y_pred=classifier.predict(x_test)


# In[23]:


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))


# In[24]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
 
cm = confusion_matrix(y_test, y_pred)
 
cm


# In[ ]:




