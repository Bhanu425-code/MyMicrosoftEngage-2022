#!/usr/bin/env python
# coding: utf-8

# IMPORT LIBRARIES

# In[1]:


import io
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# READ DATA

# In[2]:


auto_ds = pd.read_csv("price_pred.csv")
auto_ds.head()


# In[3]:


auto_ds.columns


# In[4]:


auto_ds.info(verbose=True, null_counts=True)


# In[5]:


pd.options.display.float_format="{:,.2f}".format
auto_ds.sample(n=10, random_state=0)


# DATA PREPROCESSING

# In[6]:


(auto_ds==0).sum(axis=0)


# In[7]:


auto_ds.duplicated().sum()


# In[8]:


auto_ds=auto_ds.replace('?',np.nan)


# In[9]:


auto_ds.isnull().sum()


# In[10]:


auto_ds.head()


# In[11]:


auto_ds.drop_duplicates(inplace=True)


# In[12]:


auto_ds.isnull().sum()


# In[13]:


auto_ds=auto_ds.dropna(subset=['num_of_doors','bore','stroke','horsepower','peak_rpm','price'])


# In[14]:


auto_ds.isnull().sum()


# In[15]:


auto_ds=auto_ds.replace(np.nan,0)


# In[16]:


auto_ds.head()


# In[17]:


auto_ds['normalized_losses'] = auto_ds['normalized_losses'].astype(int)


# In[18]:


auto_ds['normalized_losses']=auto_ds['normalized_losses'].replace(0,auto_ds['normalized_losses'].mean())


# In[19]:


auto_ds['normalized_losses']


# In[20]:


auto_ds.isnull().sum()


# In[21]:


auto_ds.columns


# In[22]:


auto_ds.head()


# In[23]:


auto_ds["num_of_cylinders"]


# In[24]:


def label(auto_ds, feature):
    feature_label_name = {ni: n for n, ni in enumerate(set(auto_ds[feature]))}
    return feature_label_name
symboling_label = label(auto_ds, 'symboling')
make_label = label(auto_ds, 'make')
fuel_type_label = label(auto_ds, 'fuel_type')
aspiration_label = label(auto_ds, 'aspiration')
num_of_doors_label = label(auto_ds, 'num_of_doors')
body_style_label = label(auto_ds, 'body_style')
drive_wheels_label = label(auto_ds, 'drive_wheels')
engine_location_label = label(auto_ds, 'engine_location')

engine_type_label = label(auto_ds, 'engine_type')
num_of_cylinders_label = label(auto_ds, 'num_of_cylinders')
fuel_system_label = label(auto_ds, 'fuel_system')


# In[25]:


df1 = auto_ds
df1['symboling'] = df1['symboling'].map(symboling_label)

df1['make'] = df1['make'].map(make_label)
df1['fuel_type'] = df1['fuel_type'].map(fuel_type_label)
df1['aspiration'] = df1['aspiration'].map(aspiration_label)
df1['num_of_doors'] = df1['num_of_doors'].map(num_of_doors_label)
df1['body_style'] = df1['body_style'].map(body_style_label)

df1['drive_wheels'] = df1['drive_wheels'].map(drive_wheels_label)
df1['engine_location'] = df1['engine_location'].map(engine_location_label)

df1['engine_type'] = df1['engine_type'].map(engine_type_label)
df1['num_of_cylinders'] = df1['num_of_cylinders'].map(num_of_cylinders_label)
df1['fuel_system'] = df1['fuel_system'].map(fuel_system_label)

df1


# In[26]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df2 = auto_ds
for i in df2.columns:
    df2[i] = lb.fit_transform(df2[i])
    
'''df2['Segmentation'].value_counts()'''


# DATA ANALYSIS

# In[27]:


fig, ax = plt.subplots(1, 2)
auto_ds["fuel_type"].value_counts().plot.bar(color="purple", ax=ax[0])
auto_ds["fuel_type"].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,textprops={"fontsize": 10},ax=ax[1])
fig.suptitle("fuel_type", fontsize=15)
plt.xticks(rotation=90)
plt.yticks(rotation=45)


# In[28]:


fig, ax = plt.subplots(1, 2)
auto_ds["body_style"].value_counts().plot.bar(color="purple", ax=ax[0])
auto_ds["body_style"].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,textprops={"fontsize": 10},ax=ax[1])
fig.suptitle("body_style", fontsize=15)
plt.xticks(rotation=90)
plt.yticks(rotation=45)


# In[29]:


sns.lineplot(x='symboling',y='normalized_losses',data=auto_ds)


# In[30]:


sns.lineplot(x='length',y='width',data=auto_ds)


# In[31]:


sns.lineplot(x='body_style',y='horsepower',data=auto_ds)


# In[32]:


sns.barplot(x='make',y='num_of_cylinders',data=auto_ds,hue='fuel_type')


# In[33]:


sns.barplot(x='body_style',y='aspiration',data=auto_ds,hue='num_of_doors')


# In[34]:


fig, ax = plt.subplots(1,3)
fig.suptitle("HorsePower", fontsize=15)
sns.distplot(auto_ds["horsepower"], ax=ax[0])
sns.boxplot(auto_ds["horsepower"], ax=ax[1])
sns.violinplot(auto_ds["horsepower"], ax=ax[2])


# In[35]:


fig, ax = plt.subplots(1,3)
fig.suptitle("Peak RPM", fontsize=15)
sns.distplot(auto_ds["peak_rpm"], ax=ax[0])
sns.boxplot(auto_ds["peak_rpm"], ax=ax[1])
sns.violinplot(auto_ds["peak_rpm"], ax=ax[2])


# In[36]:


fig, ax = plt.subplots(1,3)
fig.suptitle("Compression Ratio", fontsize=15)
sns.distplot(auto_ds["compression_ratio"], ax=ax[0])
sns.boxplot(auto_ds["compression_ratio"], ax=ax[1])
sns.violinplot(auto_ds["compression_ratio"], ax=ax[2])


# In[37]:


auto_ds.columns


# In[38]:


X = df1[['symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration',
       'num_of_doors', 'body_style', 'drive_wheels', 'engine_location',
       'wheel_base', 'length', 'width', 'height', ' curb_weight',
       'engine_type', 'num_of_cylinders', 'engine_size', 'fuel_system', 'bore',
       'stroke', 'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg',
       'highway_mpg']]
y = df1['price']


# TRAIN TEST SPLIT

# In[39]:


from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)


# MODELS

# In[40]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[41]:


y_pred = regressor.predict(X_test)


# In[42]:


from sklearn.metrics import r2_score, mean_squared_error
MAE = mean_squared_error(y_test, y_pred)
print('Linear Regression validation MAE = ', MAE)


# In[43]:


y_train1=y_train.copy()


# In[44]:



series_obj = pd.Series(y_train1)

# convert series object into array
arr = series_obj.values

# reshaping series
reshaped_y = arr.reshape((-1, 1))


# In[45]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_transformed = sc_X.fit_transform(X_train)
y_transformed = sc_y.fit_transform(reshaped_y)


# In[46]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_transformed, y_transformed[:135])


# In[47]:


y_pred = regressor.predict(X_test)


# In[48]:


from sklearn.metrics import r2_score, mean_squared_error
MAE = mean_squared_error(y_test, y_pred)
print('SVM validation MAE = ', MAE)


# In[49]:


from sklearn.tree import DecisionTreeRegressor
regr = DecisionTreeRegressor(max_depth=2)
regr.fit(X_train, y_train)


# In[50]:


y_pred = regressor.predict(X_test)


# In[51]:


from sklearn.metrics import r2_score, mean_squared_error
MAE = mean_squared_error(y_test, y_pred)
print('Decision Tree validation MAE = ', MAE)


# In[52]:


from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor


# In[53]:


NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(64, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


# In[54]:


NN_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2)


# In[55]:



predictions = NN_model.predict(X_test)


# In[56]:


RandomForest_model = RandomForestRegressor()
RandomForest_model.fit(X_train,y_train)

# Get the mean absolute error on the validation data
predicted = RandomForest_model.predict(X_test)
MAE = mean_absolute_error(y_test , predicted)
print('Random forest validation MAE = ', MAE)


# In[57]:


XGBModel = XGBRegressor()
XGBModel.fit(X_train,y_train , verbose=False)

# Get the mean absolute error on the validation data :
XGBpredictions = XGBModel.predict(X_test)
MAE = mean_absolute_error(y_test , XGBpredictions)
print('XGBoost validation MAE = ',MAE)


# In[58]:


model = open("regression_model_desc.pkl", "wb")
joblib.dump(XGBModel, model)
model.close()


# In[ ]:




