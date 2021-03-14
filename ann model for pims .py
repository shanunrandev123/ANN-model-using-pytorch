#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch


# In[2]:


df = pd.read_csv('diabetes.csv')


# In[3]:


df.head()


# In[4]:


df.isna().sum()


# In[7]:


# sns.pairplot(data=df, hue="SkinThickness")


# In[9]:


import numpy as np
df['Outcome'] = np.where(df['Outcome']==1,"Diabetic","notadiabetic")


# In[10]:


df.head()


# In[13]:


g=sns.pairplot(data=df, hue='BMI')
g.fig.set_size_inches(15,15)


# In[14]:


df = pd.read_csv('diabetes.csv')


# In[15]:


df.head()


# In[70]:


from sklearn.model_selection import train_test_split
X = df.drop(['Outcome'], axis = 1).values
y = df['Outcome'].values


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[21]:


# X = df.drop(['Outcome'], axis = 1).values
# y = df['Outcome'].values


# In[77]:


X


# In[78]:


y


# In[34]:





# In[79]:


X_train


# In[74]:


y_train


# In[71]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[82]:


#CREATING TENSORS AND COVERTING THE INDEPENDENT FEATURES INTO FLOAT BECAUSE IT IS COMPULSORY FOR INDEPENDENT FEATURES

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# In[83]:


df.shape


# In[100]:


class ANN_Model(nn.Module):
    def __init__(self,input_features=8,hidden1=20,hidden2=20,out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features,hidden1)
        self.f_connected2 = nn.Linear(hidden1,hidden2)
        self.out = nn.Linear(hidden2,out_features)
    def forward(self,x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x
    
    
        
        
    


# In[85]:


torch.manual_seed(20)


# In[102]:


model = ANN_Model()


# In[101]:


model.parameters


# In[58]:


#backward propagation


# In[103]:


loss_function = nn.CrossEntropyLoss()


# In[105]:


optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


# In[109]:


epochs = 500
final_losses = []
for i in range(epochs):
    i = i + 1
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred,y_train)
    final_losses.append(loss)
    if i%10==1:
        print("epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[110]:


#plot loss function


# In[111]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[114]:


plt.plot(range(epochs), final_losses)
plt.xlabel('epoch')
plt.ylabel('finalosses')


# In[120]:


predictions = []
with torch.no_grad():
    
    for i,data in enumerate(X_test):
        y_pred=model(data)
        predictions.append(y_pred.argmax().item())
        print(y_pred.argmax().item())
        
        
    
    


# In[121]:


predictions


# In[122]:


from sklearn.metrics import confusion_matrix


# In[123]:


cm = confusion_matrix(y_test,predictions)


# In[124]:


cm


# In[125]:


from sklearn.metrics import accuracy_score


# In[126]:


score = accuracy_score(predictions,y_test)


# In[127]:


score


# In[128]:


torch.save(model, 'diabetezz.pt')


# In[ ]:




