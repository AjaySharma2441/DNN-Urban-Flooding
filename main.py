#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import all the library requird
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn import metrics


# In[41]:


#Reading data for processing
os.chdir('H:/M.Tech/Workplace/NN/Patna All/Patna_Random') 
filename = 'Training.csv'
dataset = pd.read_csv(filename)
list(dataset)
dataset.head()


# In[42]:



Class_ID = dataset[['Class']]
X = dataset[['Slope','DEM','DS','DI','Drainage','GWL','DR','LULC','Pop_Density','Rainfall','DRd']]
Y= np.ravel(Class_ID)


# In[43]:


#Defining Neural Network Layer
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
input_Layer = 10
Layer1 = 200
Layer2 = 200
Layer3 = 200
Layer4 = 200
output_num_units = 5


# In[38]:


model = Sequential([
    Dense(output_dim=Layer1, 
          input_dim=input_Layer, 
          kernel_regularizer=l2(0.0001), 
          activation='relu'),
    Dropout(0.2),
    Dense(output_dim=Layer2, 
          input_dim=Layer1, 
          kernel_regularizer=l2(0.0001), 
          activation='relu'),
    Dropout(0.2),
    Dense(output_dim=Layer3, 
          input_dim=Layer2,  
          kernel_regularizer=l2(0.0001), 
          activation='relu'),
    Dropout(0.1),
    Dense(output_dim=Layer4, 
          input_dim=Layer3,  
          kernel_regularizer=l2(0.0001), 
          activation='relu'),
    Dropout(0.1),
    
    Dense (output_dim=Layer5,
          input_dim=Layer4,
          kernel_regularizer=l2(0.0001),
          activation='relu')
    Dense(output_dim=output_num_units, input_dim=Layer5, activation='softmax'),
 ])


# In[44]:


model.summary()


# In[54]:


#training and testing of data
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
history=model.fit(X_train, Y_train,epochs=100, batch_size=100, validation_split = 0.2,verbose=1,)


# In[8]:


print(history.history.keys())


# In[46]:


#accuracy graph
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()


# In[47]:


#model loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[48]:


Y_pred = model.predict(X_test)
score = model.evaluate(X_test, Y_test,batch_size=100, verbose=1)
print(score)
print("Baseline Error: %.2f%%" % (100-score[1]*100))


# In[51]:


#cunfusion matrix
test_class = model.predict_classes(X_test)
print("Confussion matrix:\n%s" %
      metrics.confusion_matrix(Y_test, test_class))


# In[52]:


#classifiaction report
print("Classification report:\n%s" %
      metrics.classification_report(Y_test, test_class))
print("Classification accuracy: %f" %
      metrics.accuracy_score(Y_test, test_class))


# In[14]:


#input data for prediction 
grid = 'Data.csv'
grid_point = pd.read_csv(grid)
X_grid = grid_point[['Slope','DEM','DS','DI','Drainage','GWL','DR','LULC','Pop_Density','Rainfall','DRd']]
xy_grid=grid_point[['POINTID','X', 'Y']]
X_grid = preprocessing.scale(X_grid)
grid_class = pd.DataFrame(model.predict_classes(X_grid))
grid_class_xy = pd.concat([xy_grid, grid_class], axis=1, join_axes=[xy_grid.index])
grid_class_xy.columns.values[3] = 'Class_ID'


# In[53]:


#Labling data
id = 'Lable.csv'
LU_ID = pd.read_csv(id)
print(LU_ID)


# In[16]:


#exporting classified data in csv format
grid_class_final=pd.merge(grid_class_xy, LU_ID, left_on='Class_ID', right_on='Class_ID', how='left')
grid_class_final.to_csv('Random2.csv', index=False)


# In[ ]:





# In[ ]:




