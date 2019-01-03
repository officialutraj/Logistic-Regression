
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('titanic_train.csv')


# In[4]:


train.head()


# In[5]:


train.info()


# In[6]:


train.isnull()


# In[8]:


sns.heatmap(train.isnull(),yticklabels = False,cbar=False ,cmap='viridis')


# In[16]:


sns.countplot(x='Survived',data=train)


# In[17]:


sns.countplot(x='Survived',hue = 'Sex',data=train)


# In[18]:


sns.countplot(x='Survived',hue = 'Pclass',data=train)


# In[21]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[24]:


#misisng data 
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass', y='Age',data=train)


# In[27]:


#fill mean of age in missing age data
def fill_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
            
        else :
            return 24
        
    else:
        return Age
    


# In[28]:


train['Age'] = train[['Age', 'Pclass']].apply(fill_age,axis = 1)


# In[30]:


sns.heatmap(train.isnull(),yticklabels = False,cbar=False ,cmap='viridis')


# In[32]:


train.drop('Cabin',inplace = True,axis = 1)


# In[33]:


sns.heatmap(train.isnull(),yticklabels = False,cbar=False ,cmap='viridis')


# In[34]:


train.head()


# In[36]:


train.dropna(inplace = True)


# In[37]:


train.info()


# In[38]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[39]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[40]:


train = pd.concat([train,sex,embark],axis=1)


# In[41]:


train.head()


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[44]:


from sklearn.linear_model import LogisticRegression


# In[45]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[46]:


predictions = logmodel.predict(X_test)


# In[47]:


from sklearn.metrics import classification_report


# In[48]:


print(classification_report(y_test,predictions))

