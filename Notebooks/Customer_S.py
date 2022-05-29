#!/usr/bin/env python
# coding: utf-8

# IMPORT LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import joblib


# READ DATA

# In[2]:


auto_ds = pd.read_csv("train2.csv")
auto_ds.head()


# In[3]:


auto_ds.columns


# In[4]:


auto_ds.info(verbose=True, null_counts=True)


# In[5]:


pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
pd.options.display.float_format="{:,.2f}".format
auto_ds.sample(n=10, random_state=0)


# DATA PRE-PROCESSING

# In[6]:


(auto_ds==0).sum(axis=0)


# In[7]:


auto_ds.duplicated().sum()


# BALANCED DATA

# In[8]:


auto_balancing = pd.DataFrame()
auto_balancing["Count"] = auto_ds["Segmentation"].value_counts()
auto_balancing["Count%"] = auto_ds["Segmentation"].value_counts()/auto_ds.shape[0]*100

auto_balancing


# In[9]:


auto_ds.describe(include="all")


# In[10]:


auto_ds.drop_duplicates(inplace=True)


# In[11]:


auto_ds.isnull().sum()


# In[12]:


len(auto_ds)


# In[13]:


auto_ds.dropna(subset=["Ever_Married"], inplace=True)
auto_ds.dropna(subset=["Graduated"], inplace=True)
auto_ds.dropna(subset=["Profession"], inplace=True)
auto_ds.dropna(subset=["Work_Experience"], inplace=True)
auto_ds.dropna(subset=["Family_Size"], inplace=True)
auto_ds.dropna(subset=["Var_1"], inplace=True)


# In[14]:


auto_ds.isnull().sum()


# In[15]:


auto_ds.drop(["ID"], axis=1, inplace=True)
auto_ds.drop(["Var_1"], axis=1, inplace=True)


# In[16]:


def label(auto_ds, feature):
    feature_label_name = {ni: n for n, ni in enumerate(set(auto_ds[feature]))}
    return feature_label_name
Gender_label = label(auto_ds, 'Gender')
Ever_Married_label = label(auto_ds, 'Ever_Married')
Age_label = label(auto_ds, 'Age')
Graduated_label = label(auto_ds, 'Graduated')
Profession_label = label(auto_ds, 'Profession')
Work_Experience_label = label(auto_ds, 'Work_Experience')
Spending_Score_label = label(auto_ds, 'Spending_Score')
Family_Size_label = label(auto_ds, 'Family_Size')
Segmentation_label = label(auto_ds, 'Segmentation')


# In[17]:


df1 = auto_ds
df1['Gender'] = df1['Gender'].map(Gender_label)
df1['Ever_Married'] = df1['Ever_Married'].map(Ever_Married_label)
df1['Age'] = df1['Age'].map(Age_label)
df1['Graduated'] = df1['Graduated'].map(Graduated_label)
df1['Profession'] = df1['Profession'].map(Profession_label)
df1['Work_Experience'] = df1['Work_Experience'].map(Work_Experience_label)
df1['Spending_Score'] = df1['Spending_Score'].map(Spending_Score_label)
df1['Family_Size'] = df1['Family_Size'].map(Family_Size_label)
df1['Segmentation'] = df1['Segmentation'].map(Segmentation_label)
df1


# In[18]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df2 = auto_ds
for i in df2.columns:
    df2[i] = lb.fit_transform(df2[i])
    
df2['Segmentation'].value_counts()


# In[19]:


df1.columns


# TRAIN TEST SPLIT

# In[20]:


X = df1[['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession','Work_Experience', 'Spending_Score', 'Family_Size']]
y = df1['Segmentation']


# In[21]:


from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)


# DATA ANALYSIS

# In[22]:


fig, ax = plt.subplots(1, 2)
auto_ds["Segmentation"].value_counts().plot.bar(color="purple", ax=ax[0])
auto_ds["Segmentation"].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,textprops={"fontsize": 10},ax=ax[1])
fig.suptitle("Segmentation Frequency", fontsize=15)
plt.xticks(rotation=90)
plt.yticks(rotation=45)


# In[23]:


fig, ax = plt.subplots(1, 2)
auto_ds["Gender"].value_counts().plot.bar(color="purple", ax=ax[0])
auto_ds["Gender"].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,textprops={"fontsize": 10},ax=ax[1])
fig.suptitle("Gender Frequency", fontsize=15)
plt.xticks(rotation=90)
plt.yticks(rotation=45)


# In[24]:


fig, ax = plt.subplots(1, 2)
auto_ds["Graduated"].value_counts().plot.bar(color="purple", ax=ax[0])
auto_ds["Graduated"].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,textprops={"fontsize": 10},ax=ax[1])
fig.suptitle("Graduation Frequency", fontsize=15)
plt.xticks(rotation=90)
plt.yticks(rotation=45)


# In[25]:


fig, ax = plt.subplots(1, 2)
auto_ds["Profession"].value_counts().plot.bar(color="purple", ax=ax[0])
auto_ds["Profession"].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,textprops={"fontsize": 10},ax=ax[1])
fig.suptitle("Profession Frequency", fontsize=15)
plt.xticks(rotation=90)
plt.yticks(rotation=45)


# In[26]:


fig, ax = plt.subplots(1, 2)
auto_ds["Spending_Score"].value_counts().plot.bar(color="purple", ax=ax[0])
auto_ds["Spending_Score"].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,textprops={"fontsize": 10},ax=ax[1])
fig.suptitle("Spending Score Frequency", fontsize=15)
plt.xticks(rotation=90)
plt.yticks(rotation=45)


# In[27]:


sns.barplot(x='Segmentation',y='Age',data=auto_ds,hue='Gender')


# In[28]:


fig, ax = plt.subplots(1,3)
fig.suptitle("Age Distribution", fontsize=15)
sns.distplot(auto_ds["Age"], ax=ax[0])
sns.boxplot(auto_ds["Age"], ax=ax[1])
sns.violinplot(auto_ds["Age"], ax=ax[2])


# In[29]:


fig, ax = plt.subplots(1,3)
fig.suptitle("Work Experience Distribution", fontsize=15)
sns.distplot(auto_ds["Work_Experience"], ax=ax[0])
sns.boxplot(auto_ds["Work_Experience"], ax=ax[1])
sns.violinplot(auto_ds["Work_Experience"], ax=ax[2])


# In[30]:


fig, ax = plt.subplots(1,3)
fig.suptitle("Family Size Distribution", fontsize=15)
sns.distplot(auto_ds["Family_Size"], ax=ax[0])
sns.boxplot(auto_ds["Family_Size"], ax=ax[1])
sns.violinplot(auto_ds["Family_Size"], ax=ax[2])


# MODELS

# In[31]:


def classifier(model, X_train, X_test, y_train, y_test):
    clf = model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score, model


# In[32]:


score, lr_model = classifier(LogisticRegression(), X_train, X_test, y_train, y_test)
print('LogisticRegression score = ',score)


# In[33]:


score, rf = classifier(RandomForestClassifier(n_estimators=50), X_train, X_test, y_train, y_test)
print('Random Forest Classifer score = ',score)


# In[34]:


paramgrid = {'max_depth': list(range(1,20,2)),'n_estimators':list(range(1,200,20))}
grid_search = GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)


# In[35]:


grid_search.fit(X_train, y_train)
grid_search.best_estimator_


# In[36]:


score, rf = classifier(RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                              class_weight=None,
                                              criterion='gini', max_depth=None,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              max_samples=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators=100, n_jobs=None,
                                              oob_score=False, random_state=1,
                                              verbose=0, warm_start=False), X_train, X_test, y_train, y_test)
print('Random Forest Classifer(GridSearchCV) score = ',score)


# In[37]:


from sklearn import tree
#import graphviz
tree_clf = tree.DecisionTreeClassifier(max_depth = 4)
tree_clf.fit(X_train,y_train)
y_pred = tree_clf.predict(X_test)
print('Decision Tree accuracy score = ',accuracy_score(y_test, y_pred))
#dot_data = tree.export_graphviz(tree_clf,feature_names = X.columns.tolist())
#graph = graphviz.Source(dot_data)
#graph


# In[38]:


score, nn = classifier(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,4)), X_train, X_test, y_train, y_test)
print('MLPClassifier score= ',score)


# In[39]:


from xgboost import XGBRegressor
XGBModel = XGBRegressor()
XGBModel.fit(X_train,y_train , verbose=False)

# Get the mean absolute error on the validation data :
XGBpredictions = XGBModel.predict(X_test)
print('XGBRegressor score= ',accuracy_score(y_test, y_pred))


# In[40]:


tree_model2 = open("descision_tree_model.pkl", "wb")
joblib.dump(tree_clf, tree_model2)
tree_model2.close()


# In[ ]:





# In[ ]:




