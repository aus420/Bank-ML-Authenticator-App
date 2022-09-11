#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st


# In[2]:


df=pd.read_csv(r"C:\Users\AbhishekDas\Downloads\BankNote_Authentication.csv")


# In[3]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[4]:


X.head()


# In[5]:


y.head()


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=0)


# In[8]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(xtrain,ytrain)


# In[9]:


y_pred=classifier.predict(xtest)


# In[10]:


from sklearn.metrics import accuracy_score
score=accuracy_score(ytest,y_pred)


# In[11]:


score


# In[12]:


import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


# In[13]:


classifier.predict([[2,3,4,1]])


# In[14]:


from PIL import Image


# In[15]:


pickle_in=open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


# In[16]:


def Welcome():
    return "Welcome All"


# In[17]:


def predict_note_authentication(variance,skewness,curtosis,entropy):
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction


# In[18]:


def main():
    st.title("Bank Authenticator")
    html_temp="""
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Bank Authenticator ML App</h2>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    variance=st.text_input("Variance","Type Here")
    skewness=st.text_input("Skewness","Type Here")
    curtosis=st.text_input("curtosis","Type Here")
    entropy=st.text_input("entropy","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(variance,skewness,curtosis,entropy)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")


# In[ ]:


if __name__=='__main__':
    main()

