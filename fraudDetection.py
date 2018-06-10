
# coding: utf-8

# In[2]:


import pandas as pd

# Read csv data
dataframe = pd.read_csv('creditcard.csv')


# In[52]:


#Display Data
dataframe.head()
dataframe.describe()


# In[ ]:


#The data contains 28 features and the amount of each transaction
#We need to normalise the Amount as it has a very high maximum value compared to other features
#Ignoring the time feature for now.Although it may help point out fraudulent transactions which may be closer to each other


# In[39]:


#Importing Dependencies
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize


# In[26]:


#Separating Features and Targets
X = dataframe.iloc[:,1:31]
Y = dataframe.iloc[:,-1]


# In[28]:


#Normalizing the features
X = normalize(X)


# In[40]:


#Using SVM Classifier
model = SVC()


# In[35]:


#Splitting into train and test set.Using starify to make sure class variables have same distribution in train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,stratify=Y,test_size=0.5,random_state=0)


# In[46]:


model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)


# In[49]:


Y_score = model.decision_function(X_test)


# In[47]:


print(classification_report(Y_test,Y_pred))


# In[48]:


#Plotting Precision Recall Curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


# In[51]:


precision,recall,_=precision_recall_curve(Y_test,Y_score)
plt.step(recall,precision,color='b',alpha=0.2,where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

