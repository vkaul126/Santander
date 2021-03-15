#!/usr/bin/env python
# coding: utf-8

# In[60]:



   #Loading Libraries:-
    
#pip install eli5

import os
import numpy as np
import pandas as pd
import seaborn as sns
import eli5
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.inspection import plot_partial_dependence as pdp
from pdpbox import pdp, get_dataset, info_plots
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,classification_report,roc_curve,auc


random_state=42
np.random.seed(random_state)
import warnings
warnings.filterwarnings('ignore')


# In[61]:


os.chdir("C:/Users/Vaishali/Documents/Santander")


# In[62]:


os.getcwd()


# In[63]:


df_train=pd.read_csv("train.csv")
pd.options.display.max_columns = None


# In[64]:


df_train.head()


# In[65]:



#Shape of the dataset
df_train.shape


# In[66]:


#Target Class Count
target_class=df_train['target'].value_counts()
print('Count of the target class :\n',target_class)

#Percentage of target class count
per_target_class=df_train['target'].value_counts()/len(df_train)*100
print('Percentage of target class count :\n',per_target_class)


# In[67]:


#Count plot & violin plot for target class
fig,ax=plt.subplots(1,2,figsize=(20,5))
sns.countplot(df_train.target.values,ax=ax[0],palette='spring')
sns.violinplot(x=df_train.target.values,y=df_train.index.values,ax=ax[1],palette='spring')
sns.stripplot(x=df_train.target.values,y=df_train.index.values,jitter=True,color='black',linewidth=0.5,size=0.5,alpha=0.5,ax=ax[1],palette='spring')
ax[0].set_xlabel('Target')
ax[1].set_xlabel('Target')
ax[1].set_ylabel('Index')


# In[68]:


get_ipython().run_cell_magic('time', '', '\n#Distribution of train attributes-\n\ndef plot_train_attribute_distribution(t0,t1,label1,label2,train_attributes):\n    i=0\n    sns.set_style(\'darkgrid\')\n    \n    fig=plt.figure()\n    ax=plt.subplots(10,10,figsize=(22,18))\n    \n    for attribute in train_attributes :\n        i+=1\n        plt.subplot(10,10,i)\n        sns.distplot(t0[attribute],hist=False,label=label1)\n        sns.distplot(t1[attribute],hist=False,label=label2)\n        plt.legend()\n        plt.xlabel(\'Attribute\',)\n        sns.set_style("ticks",{"xtick.major.size": 8, "ytick.major.size": 8})\n    plt.show()')


# In[69]:


#Observing first 100 train attributes


#Corresponding to negative class-
t0=df_train[df_train.target.values==0]

#Corresponding to possitive class-
t1=df_train[df_train.target.values==1]

#train attributes from 2 to 102 -
train_attributes=df_train.columns.values[2:102]

#Plot distribution of train attributes-
plot_train_attribute_distribution(t0,t1,'0','1',train_attributes)


# In[70]:


#train attributes from 102 to 202 -
train_attributes=df_train.columns.values[102:202]

#Plot distribution of train attributes-
plot_train_attribute_distribution(t0,t1,'0','1',train_attributes)


# In[71]:



#Importing the test dataset:-
df_test=pd.read_csv("test.csv")


# In[72]:


df_test.head()


# In[73]:


#Shape of the dataset-
df_test.shape


# In[75]:



#Distribution of test attributes-

#Distribution of test attributes-

def plot_test_attribute_distribution(test_attributes):
    i=0
    sns.set_style('darkgrid')
    
    fig=plt.figure()
    ax=plt.subplots(10,10,figsize=(22,18))
    
    for attribute in test_attributes:
        i+=1
        plt.subplot(10,10,i)
        sns.distplot(df_test[attribute],hist=False)
        plt.xlabel('Attribute',)
        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    plt.show()




# In[76]:


#test attribiutes from 1 to 101 -
test_attributes=df_test.columns.values[1:101]

#Plot distribution of test attributes -
plot_test_attribute_distribution(test_attributes)


# In[77]:


#test attributes from 101 to 202-
test_attributes=df_test.columns.values[101:202]

#Plot the distribution of test attributes-
plot_test_attribute_distribution(test_attributes)


# In[78]:


#Distribution of Mean Values per column in train & test dataset:-
plt.figure(figsize=(16,8))

#Train attributes-
train_attributes=df_train.columns.values[2:202]

#Test attributes-
test_attributes=df_test.columns.values[1:201]

#Distribution plot for mean values per column in train attributes:
sns.distplot(df_train[train_attributes].mean(axis=0),color='red',kde=True,bins=150,label='train')

#Distribution plot for mean values per column in test attributes:
sns.distplot(df_test[test_attributes].mean(axis=0),color='blue',kde=True,bins=150,label='test')

plt.title('Distribution of Mean Values per column in train & test dataset')
plt.legend()
plt.show()


#Distribution of Mean Values per column in train & test dataset:-
plt.figure(figsize=(16,8))

#Distribution plot for mean values per rows in train attributes:
sns.distplot(df_train[train_attributes].mean(axis=1),color='red',kde=True,bins=150,label='train')

#Distribution plot for mean values per rows in test attributes:
sns.distplot(df_test[test_attributes].mean(axis=1),color='blue',kde=True,bins=150,label='test')

plt.title('Distribution of Mean Values per row in train & test dataset')
plt.legend()
plt.show()


# In[79]:


#Distribution of S.D Values per column in train & test dataset:-
plt.figure(figsize=(16,8))

#Train attributes-
train_attributes=df_train.columns.values[2:202]

#Test attributes-
test_attributes=df_test.columns.values[1:201]

#Distribution plot for S.D values per column in train attributes:
sns.distplot(df_train[train_attributes].std(axis=0),color='blue',kde=True,bins=150,label='train')

#Distribution plot for S.D values per column in test attributes:
sns.distplot(df_test[test_attributes].std(axis=0),color='green',kde=True,bins=150,label='test')

plt.title('Distribution of S.D Values per column in train & test dataset')
plt.legend()
plt.show()


#Distribution of S.D Values per column in train & test dataset:-
plt.figure(figsize=(16,8))

#Distribution plot for S.D values per rows in train attributes:
sns.distplot(df_train[train_attributes].std(axis=1),color='blue',kde=True,bins=150,label='train')

#Distribution plot for S.D values per rows in test attributes:
sns.distplot(df_test[test_attributes].std(axis=1),color='green',kde=True,bins=150,label='test')

plt.title('Distribution of S.D Values per row in train & test dataset')
plt.legend()
plt.show()


# In[20]:



get_ipython().run_cell_magic('time', '', "#Distribution of skew Values per column in train & test dataset:-\nplt.figure(figsize=(16,8))\n\n#Train attributes-\ntrain_attributes=df_train.columns.values[2:202]\n\n#Test attributes-\ntest_attributes=df_test.columns.values[1:201]\n\n#Distribution plot for skew values per column in train attributes:\nsns.distplot(df_train[train_attributes].skew(axis=0),color='red',kde=True,bins=150,label='train')\n\n#Distribution plot for skew values per column in test attributes:\nsns.distplot(df_test[test_attributes].skew(axis=0),color='green',kde=True,bins=150,label='test')\n\nplt.title('Distribution of skewness Values per column in train & test dataset')\nplt.legend()\nplt.show()\n\n\n#Distribution of skew Values per column in train & test dataset:-\nplt.figure(figsize=(16,8))\n\n#Distribution plot for skew values per rows in train attributes:\nsns.distplot(df_train[train_attributes].skew(axis=1),color='red',kde=True,bins=150,label='train')\n\n#Distribution plot for skew values per rows in test attributes:\nsns.distplot(df_test[test_attributes].skew(axis=1),color='green',kde=True,bins=150,label='test')\n\nplt.title('Distribution of skewness Values per row in train & test dataset')\nplt.legend()\nplt.show()")


# In[21]:


#Distribution of kurtosis Values per column in train & test dataset:-
plt.figure(figsize=(16,8))

#Train attributes-
train_attributes=df_train.columns.values[2:202]

#Test attributes-
test_attributes=df_test.columns.values[1:201]

#Distribution plot for kurtosis values per column in train attributes:
sns.distplot(df_train[train_attributes].kurtosis(axis=0),color='red',kde=True,bins=150,label='train')

#Distribution plot for kurtosis values per column in test attributes:
sns.distplot(df_test[test_attributes].kurtosis(axis=0),color='blue',kde=True,bins=150,label='test')

plt.title('Distribution of kurtosis Values per column in train & test dataset')
plt.legend()
plt.show()


#Distribution of kurtosis Values per column in train & test dataset:-
plt.figure(figsize=(16,8))

#Distribution plot for kurtosis values per rows in train attributes:
sns.distplot(df_train[train_attributes].kurtosis(axis=1),color='red',kde=True,bins=150,label='train')

#Distribution plot for kurtosis values per rows in test attributes:
sns.distplot(df_test[test_attributes].kurtosis(axis=1),color='green',kde=True,bins=150,label='test')

plt.title('Distribution of kurtosis Values per row in train & test dataset')
plt.legend()
plt.show()


# In[22]:


#Finding the missing values in train & test dataset:-
train_missing=df_train.isnull().sum().sum()
test_missing=df_test.isnull().sum().sum()

print('Missing values in train data:',train_missing)
print('Missing values in test data:',test_missing)


# In[23]:


#Correlation in train attiributes-
train_attributes=df_train.columns.values[2:202]
train_correlation=df_train[train_attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()
train_correlation=train_correlation[train_correlation['level_0']!=train_correlation['level_1']]
print(train_correlation.head(10))
print(train_correlation.tail(10))


# In[24]:


#Correlation in test attiributes-
test_attributes=df_test.columns.values[1:201]
test_correlation=df_test[train_attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()
test_correlation=test_correlation[test_correlation['level_0']!=test_correlation['level_1']]
print(test_correlation.head(10))
print(test_correlation.tail(10))


# In[25]:


train_correlation=df_train[train_attributes].corr()
train_correlation=train_correlation.values.flatten()
train_correlation=train_correlation[train_correlation!=1]


test_correlation=df_test[test_attributes].corr()
test_correlation=test_correlation.values.flatten()
test_correlation=test_correlation[test_correlation!=1]

plt.figure(figsize=(20,5))
sns.distplot(train_correlation,color="blue",label="train")
sns.distplot(test_correlation,color="red",label="test")
plt.xlabel("Correlation values found in train & test data")
plt.ylabel("Density")
plt.title ("Correlation values in train & test data")
plt.legend()


# In[26]:


#Feature Engineering :- Performing feature engineering by using-

#- Permutation Importance
#- Partial dependence plots

#Training & testing data:
X=df_train.drop(columns=['ID_code','target'],axis=1)
test=df_test.drop(columns=['ID_code'],axis=1)
y=df_train['target']


# In[27]:


#Building a simple model to find the features which are more important:
#Split the train data:-
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)


# In[28]:


#Random Forest Classifier:-

rf_model=RandomForestClassifier(n_estimators=10,random_state=42)

#fitting the model:-
rf_model.fit(X_test,y_test)


# In[29]:


#Permutation Importance:-
from eli5.sklearn import PermutationImportance
perm_imp=PermutationImportance(rf_model,random_state=42)

#fitting the model:-
perm_imp.fit(X_test,y_test)


# In[30]:


#Important Features:-
eli5.show_weights(perm_imp,feature_names=X_test.columns.tolist(),top=200)


# In[31]:


#Calculation of partial dependence plots on random forest:-
#we are observing impact of main features which are discovered in previous section by using PDP Plot.


features=[v for v in X_test.columns if v not in ['ID_code','target']]
pdp_data=pdp.pdp_isolate(rf_model, dataset=X_test, model_features=features, feature='var_6')


# In[32]:


#Plot feature for var_6:-
pdp.pdp_plot(pdp_data,'var_6')
plt.show()


# In[80]:


#Plot feature for var_53:-
features=[v for v in X_test.columns if v not in ['ID_code','target']]
pdp_data=pdp.pdp_isolate(rf_model, dataset=X_test, model_features=features, feature='var_53')
pdp.pdp_plot(pdp_data,'var_53')
plt.show()


# In[35]:


#Spliting the data via Sratified KFold Cross Validator:-

#Training Data:
X=df_train.drop(['ID_code','target'],axis=1)
Y=df_train['target']

#Stratified KFold Cross Validator:-
skf=StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
for train_index, valid_index in skf.split(X,Y): 
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index] 
    y_train, y_valid = Y.iloc[train_index], Y.iloc[valid_index]
    
print('Shape of X_train :',X_train.shape)
print('Shape of X_valid :',X_valid.shape)
print('Shape of y_train :',y_train.shape)
print('Shape of y_valid :',y_valid.shape)


# In[36]:


#Logistic Regression Model:-
lr_model=LogisticRegression(random_state=42)
#fitting the model-
lr_model.fit(X_train,y_train)
    


# In[ ]:


#Accuracy of model-
lr_score=lr_model.score(X_train,y_train)
print('Accuracy of lr_model :',lr_score)


# In[41]:


#Cross validation prediction of lr_model-
cv_predict=cross_val_predict(lr_model,X_valid,y_valid,cv=5)
#Cross validation score-
cv_score=cross_val_score(lr_model,X_valid,y_valid,cv=5)
print('cross val score :',np.average(cv_score))


# In[81]:


#Confusion matrix:-
cm=confusion_matrix(y_valid,cv_predict)
cm=pd.crosstab(y_valid,cv_predict)
cm


# In[42]:


#ROC_AUC SCORE:-
roc_score=roc_auc_score(y_valid,cv_predict)
print('ROC Score:',roc_score)


# In[43]:


#ROC_AUC_Curve:-
plt.figure()
false_positive_rate,recall,thresholds=roc_curve(y_valid,cv_predict)
roc_auc=auc(false_positive_rate,recall)
plt.title('Reciver Operating Characteristics(ROC)')
plt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall(True Positive Rate)')
plt.xlabel('False Positive Rate')
plt.show()
print('AUC:',roc_auc)


# In[44]:


#Classification report:-
classification_scores=classification_report(y_valid,cv_predict)
print(classification_scores)


# In[45]:


#Model performance on test data:-
X_test=df_test.drop(['ID_code'],axis=1)
lr_pred=lr_model.predict(X_test)
print(lr_pred)


# In[94]:


from imblearn.over_sampling import SMOTE
#SMOTE:-
sm = SMOTE(random_state=42, sampling_strategy=1.0)
#Generating synthetic data points
X_smote,y_smote=sm.fit_sample(X_train,y_train)
X_smote_v,y_smote_v=sm.fit_sample(X_valid,y_valid)


# In[95]:


#Building Logistsic regression model on synthetic data points:-
#Logistic regression model for SMOTE:-
smote=LogisticRegression(random_state=42)
#fitting the smote model:-
smote.fit(X_smote,y_smote)


# In[96]:


#Accuracy of the model:-
smote_score=smote.score(X_smote,y_smote)
print('Accuracy of the smote_model :',smote_score)


# In[49]:


#Cross validation prediction for SMOTE:-
cv_pred=cross_val_predict(smote,X_smote_v,y_smote_v,cv=5)
#Cross validation score:-
cv_score=cross_val_score(smote,X_smote_v,y_smote_v,cv=5)
print('Cross validation score :',np.average(cv_score))


# In[50]:


#Confusion matrix:-
cm=confusion_matrix(y_smote_v,cv_pred)
cm=pd.crosstab(y_smote_v,cv_pred)


# In[51]:


cm


# In[87]:


#ROC_AUC SCORE:-

roc_score=roc_auc_score(y_smote_v,cv_pred)
print('ROC score:',roc_score)
#sklearn.metrics.roc_auc_score()


# In[88]:



#ROC_AUC Curve:-
plt.figure()
false_positive_rate,recall,thresholds=roc_curve(y_smote_v,cv_pred)
roc_auc=auc(false_positive_rate,recall)
plt.title('Reciver Operating Characteristics(ROC)')
plt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall(True Positive Rate)')
plt.xlabel('False Positive Rate')
plt.show()
print('AUC:',roc_auc)


# In[53]:


#Classification Report:-
scores=classification_report(y_smote_v,cv_pred)
print(scores)


# In[97]:


#Predicting the model-
X_test=df_test.drop(['ID_code'],axis=1)
smote_pred=smote.predict(X_test)
print(smote_pred)


# In[90]:


lbg_train=lbg.Dataset(X_train,label=y_train)

#Validation data-
lbg_valid=lbg.Dataset(X_valid,label=y_valid)


# In[91]:



#Selecting best hyperparameters by tuning of different parameters:-
params={'boosting_type': 'gbdt', 
          'max_depth' : -1, #no limit for max_depth if <0
          'objective': 'binary',
          'boost_from_average':False, 
          'nthread': 20,
          'metric':'auc',
          'num_leaves': 50,
          'learning_rate': 0.01,
          'max_bin': 100,      #default 255
          'subsample_for_bin': 100,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'bagging_fraction':0.5,
          'bagging_freq':5,
          'feature_fraction':0.08,
          'min_split_gain': 0.45, #>0
          'min_child_weight': 1,
          'min_child_samples': 5,
          'is_unbalance':True,
          }


# In[57]:


#Training lgbm model:-
num_rounds=10000
lgbm= lbg.train(params,lbg_train,num_rounds,valid_sets=[lbg_train,lbg_valid],verbose_eval=1000,early_stopping_rounds = 5000)
lgbm


# In[58]:


#LGBM model performance on test data:-
X_test=df_test.drop(['ID_code'],axis=1)
#Predict the model:-

#probability predictions
lgbm_predict_prob=lgbm.predict(X_test,random_state=42,num_iteration=lgbm.best_iteration)

#Convert to binary output 1 or 0
lgbm_predict=np.where(lgbm_predict_prob>=0.5,1,0)
print(lgbm_predict_prob)
print(lgbm_predict)


# In[59]:


#Plotting of important Features:-
lbg.plot_importance(lgbm,max_num_features=50,importance_type="split",figsize=(20,50))


# In[92]:


#Final submission:-
df_sub=pd.DataFrame({'ID_code':df_test['ID_code'].values})
df_sub['lgbm_predict_prob']=lgbm_predict_prob
df_sub['lgbm_predict']=lgbm_predict
df_sub.to_csv('submission.csv',index=False)
df_sub.head()


# In[ ]:





# In[ ]:




