
# coding: utf-8

# Ensemble classifiers with best F1 value:

# In[1]:

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from matplotlib.pylab import plt
import time
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


# Preprocess Data

# In[2]:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_excel("default_of_credit_card_clients.xls",header=1,skiprows = 0,index_col = 0)
data.rename(index = str, columns = {'default payment next month': 'lable'},inplace = True)

categorical = ['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
non_categorical = [x for x in data.columns if x not in categorical]

data = pd.get_dummies(data, columns = categorical)

categorical = [x for x in data.columns if x not in non_categorical]
non_categorical.remove("lable")

train, test = train_test_split(data, test_size = 0.2, random_state = 888)
y_train = train['lable']
y_test = test['lable']
train = train.drop(['lable'],axis = 1)
test = test.drop(['lable'],axis = 1)

scaler = StandardScaler(copy = False, with_mean=True, with_std=True)
train_scaled = pd.DataFrame(scaler.fit_transform(train[non_categorical]), columns = non_categorical,index = train.index)
test_scaled = pd.DataFrame(scaler.transform(test[non_categorical]),columns = non_categorical, index = test.index)
X_train = pd.merge(train_scaled, train[categorical], left_index=True, right_index=True, copy = False )
X_test = pd.merge(test_scaled, test[categorical], left_index=True, right_index=True, copy = False )


# Metrics

# In[3]:

def get_mymetrix(y,y_pred, model):
    mymetrics = pd.DataFrame(data={ 
                     '0_model': [model],
                     '1_precision_score': [metrics.precision_score(y_test, y_pred)],
                     '2_recall_score': [metrics.recall_score(y_test, y_pred)],
                     '3_f1_score': [metrics.f1_score(y_test, y_pred)],
                     '4_accuracy':[metrics.accuracy_score(y_test, y_pred)]},
                   )  
    return mymetrics


# Naive Bayes:

# In[4]:

gnb = GaussianNB()
gnb.fit(X_train[non_categorical], y_train)
gnb_y_predict_proba = gnb.predict_proba(X_test[non_categorical])
mnb = MultinomialNB()
dummy_categorical = [x for x in X_train.columns if x not in non_categorical]
mnb.fit(X_train[dummy_categorical], y_train)
mnb_y_predict_proba = mnb.predict_proba(X_test[dummy_categorical])
y_predict_proba = gnb_y_predict_proba * mnb_y_predict_proba
nb_y_predict_proba = y_predict_proba
y_predict = np.zeros(len(y_predict_proba))

for i in range(len(y_predict_proba)):
    if y_predict_proba[i][0] >= y_predict_proba[i][1]:
        y_predict[i] = 0
    else:
        y_predict[i] = 1
nb_y_predict = y_predict
get_mymetrix(y_test, nb_y_predict, 'NB')


# Logistic Regression：

# In[5]:

logis = LogisticRegression(max_iter=3, class_weight='balanced')
logis.fit(X_train, y_train)
y_predict = logis.predict(X_test)
lr_y_predict = y_predict
get_mymetrix(y_test, lr_y_predict, 'LogisticRegression')


# MLP：

# In[6]:

mlp = MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=20, alpha=0.0001,
                     solver='sgd', activation='relu', learning_rate_init=1, random_state=21,tol=0.000000001)  #verbose=10,  
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
mlp_relu_y_predict = y_pred
get_mymetrix(y_test, mlp_relu_y_predict, 'MLPClassifier')


# Decision_Tree：

# In[7]:

dc = DecisionTreeClassifier(max_depth=6,min_samples_leaf = 10, class_weight={0:1,1:4})
dc.fit(X_train, y_train)
y_pred = dc.predict(X_test)
dc_y_predict = y_pred
get_mymetrix(y_test, dc_y_predict, 'DecisionTree')


# Random - Forest

# In[8]:

rf = RandomForestClassifier(max_depth=10, min_samples_leaf=5, n_estimators=80, class_weight={0:1,1:4})
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf_y_predict = y_pred
get_mymetrix(y_test, rf_y_predict, 'RandomForestTree')


# SVM:

# In[9]:

svm = LinearSVC(C=6, class_weight={0:1,1:4})
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
svm_y_predict = y_pred
get_mymetrix(y_test, svm_y_predict, 'SVM')


# Deep Learning:

# In[10]:

model = Sequential()
model.add(Dense(30, input_dim=(X_train.shape[1]), activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1,  activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(np.array(X_train), np.array(y_train), batch_size=25, epochs=5, class_weight = {0:0.2, 1:0.8})
y_test_pred = model.predict_classes(np.array(X_test))[:,0]
keras_y_predict = y_test_pred
get_mymetrix(y_test, keras_y_predict, 'DeepLearning')


# XGB Classifier:

# In[11]:

xgb = XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=140, scale_pos_weight=4)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
xgb_y_predict = y_pred
get_mymetrix(y_test, xgb_y_predict, 'XGBClassifier')


# In[12]:

all_predicts = np.stack((nb_y_predict, 
                         lr_y_predict, 
                         mlp_relu_y_predict, 
                         dc_y_predict,
                         rf_y_predict,
                         svm_y_predict,
                         keras_y_predict,
                         xgb_y_predict))


# In[13]:

all_predicts = all_predicts.astype(int)


# In[14]:

maj = np.asarray([np.argmax(np.bincount(all_predicts[:,c])) for c in range(all_predicts.shape[1])])


# In[15]:

get_mymetrix(y_test, maj, 'Ensemble')


# In[ ]:




# In[ ]:



