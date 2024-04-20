#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  # for data analysis and operating numbrical tables and time series


# In[2]:


import numpy as np # Python library to work with arrays
import seaborn as sns # Used for data exploration and data visualisation


# In[3]:


import matplotlib.pyplot as plt  # Data visulasisation and graphical charting 


# In[4]:


from sklearn.model_selection import train_test_split # Train test split 


# In[5]:


from sklearn.metrics import accuracy_score # ratio od +ve to -ve


# In[6]:


from sklearn.metrics import classification_report 


# In[7]:


import re 


# In[8]:


import string # to manipulate string data


# In[9]:


import pandas as pd

# Read Fake.csv
data_fake = pd.read_csv("C:/Users/jatin/Downloads/Fake.csv")

# Read True.csv
data_true = pd.read_csv("C:/Users/jatin/Downloads/True.csv")


# In[10]:


data_fake.head()


# In[11]:


data_true.head()


# In[12]:


data_fake["class"] = 0 
data_true["class"] = 1 


# In[13]:


data_fake.shape


# In[14]:


data_true.shape


# In[15]:


data_fake_manual_testing = data_fake.tail (10)
for i in range (23480 ,23470 , -1 ):
    data_fake.drop([i], axis = 0, inplace = True )
    
data_true_manual_testing = data_true.tail (10)
for i in range (21406 ,221416 , -1 ):
    data_fake.drop([i], axis = 0, inplace = True )


# In[16]:


data_fake.shape , data_true.shape


# In[17]:


data_fake_manual_testing['class'] = 0 
data_true_manual_testing['class'] = 1 


# In[18]:


data_fake_manual_testing.head(10)


# In[19]:


data_true_manual_testing.head(10)


# In[20]:


data_merge =  pd.concat ([data_fake , data_true], axis  = 0 )


# In[21]:


data_merge.shape


# In[22]:


data_merge.head(10)


# In[23]:


data_merge.columns


# In[24]:


data = data_merge.drop (['title' , 'date' , 'subject'] ,axis = 1)


# In[25]:


data.isnull().sum()


# In[26]:


data.shape


# In[27]:


data = data.sample (frac = 1) # for random shuffeling


# In[28]:


data.head()


# In[29]:


data.reset_index (inplace  = True)


# In[30]:


data.drop (['index'] , axis  = 1 , inplace  =True )


# In[31]:


data.columns


# In[32]:


data.head()


# In[33]:


def wordopt (text):
    text = text.lower()
    text = re.sub ('\[.*?]','',text)
    text = re.sub("//W" ," " ,text)
    text = re.sub('https?://\S+|www\.\S+', '' ,text)
    text = re.sub('<.*?>+' , '' , text)
    text = re.sub('[%s]' %re.escape(string.punctuation) , '' , text)
    text = re.sub ('\n','',text)
    text = re.sub ('\w*\d\w*' , '' ,text)
    return text 


# In[34]:


data['text'] = data['text'].apply(wordopt)


# In[35]:


x = data['text']
y = data['class']


# In[36]:


x_train , x_test ,y_train ,y_test = train_test_split (x,y,test_size = 0.25) 


# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
xv_train = vectorizer.fit_transform(x_train)

# Transform the test data using the same vectorizer
xv_test = vectorizer.transform(x_test)


# In[38]:


from sklearn .linear_model import LogisticRegression 


# In[39]:


LR = LogisticRegression()
LR.fit(xv_train, y_train)


# In[40]:


pred_lr = LR.predict(xv_test)


# In[41]:


LR.score(xv_test , y_test)


# In[42]:


print(classification_report(y_test,pred_lr))


# In[43]:


from sklearn.tree import DecisionTreeClassifier


# In[44]:


DT= DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[45]:


pred_DT = DT.predict(xv_test)


# In[46]:


DT.score(xv_test , y_test)


# In[47]:


print(classification_report(y_test,pred_DT))


# In[48]:


from sklearn.ensemble import GradientBoostingClassifier

# Initialize GradientBoostingClassifier
GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train,y_train)


# In[49]:


predict_gb = GB.predict(xv_test)


# In[50]:


GB.score (xv_test ,y_test)


# In[51]:


from sklearn.metrics import classification_report

# Assuming y_test is your true labels and pred_GB is your predicted labels
print(classification_report(y_test, predict_gb))


# In[52]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state=0)


# In[53]:


RF.fit (xv_train , y_train )


# In[54]:


pred_rf = RF.predict (xv_test)


# In[55]:


RF.score (xv_test ,y_test)


# In[56]:


print (classification_report(y_test ,pred_rf))


# In[57]:


import pandas as pd

def output_label(n):
    if n == 0:
        return "Fake news"
    elif n == 1:
        return "Not A Fake news"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    # Assuming wordopt is a function to process the text (not defined in the provided code)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)  # You need to define wordopt
    new_xv_test = vectorizer.transform(new_def_test["text"])
    
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    print("\n\nLR Prediction: {} \nDT Prediction: {} \nGB Prediction: {} \nRF Prediction: {}".format(output_label(pred_LR[0]), output_label(pred_DT[0]), output_label(pred_GB[0]), output_label(pred_RF[0])))




# In[ ]:


model.save("model.h5")


# In[ ]:


news = str(input())
manual_testing(news)


# In[ ]:


keras take h5 otherwise .pkl integrate with streamlit library


# In[ ]:





# In[ ]:





# In[ ]:




