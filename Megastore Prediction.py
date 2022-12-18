#!/usr/bin/env python
# coding: utf-8

# # Importing the Relevant Libraries

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection  import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# # Data Inspection

# In[2]:


df = pd.read_csv("C:\\Users\\khush\\OneDrive\\Documents\\Data Science\\MegaStore.csv")


# In[3]:


df.shape


# __MegaStore has 8523 rows and 12 columns__

# In[4]:


#ratio of null values
df.isnull().sum()/df.shape[0]*100


# In[5]:


#categorical features
categorical = df.select_dtypes(include =[np.object])
print("Categorical Features:",categorical.shape[1])

numerical = df.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features:",numerical.shape[1])


# __Weight__

# In[6]:


plt.figure(figsize=(8,5))
sns.boxplot('Weight',data=df)


# __The Box-Plots above clearly show no "outliers" and hence we can input the missing values with "Means"__

# In[7]:


df['Weight'] = df['Weight'].fillna(df['Weight'].mean())


# In[8]:


df['Weight'].isnull().sum()


# __Outlet Size__

# In[9]:


df['OutletSize'].isnull().sum()


# In[10]:


print(df['OutletSize'].value_counts())


# __Since the outlet_size is a categorical column. we can inpute the missing value with "Mode"__

# __Let us see its association with OutletType columns__

# In[11]:


mode_of_outletSize =df.pivot_table(values='OutletSize', columns='OutletType', aggfunc=(lambda x:x.mode()[0]))
print(mode_of_outletSize)


# In[12]:


missing_value=df['OutletSize'].isnull()
missing_value


# In[13]:


df.loc[missing_value,'OutletSize']=df.loc[missing_value,'OutletType'].apply(lambda x: mode_of_outletSize[x])


# In[14]:


df['OutletSize'].isnull().sum()


# In[15]:


df.describe()


# In[16]:


df.columns


# In[17]:


df['FatContent'].value_counts()


# __Let us fix these irregularities__

# In[18]:


df['FatContent'].replace(['low fat', 'Low','reg'],['Low Fat', 'Low Fat','Regular'], inplace=True)


# In[19]:


df['FatContent'] = df['FatContent'].astype(str)


# __Determine the year of operation of a store__

# In[20]:


df['OutletYears'] = 2021 - df['OutletEstablishmentYear']


# __Create a board category of Type of Item__

# In[21]:


df['ItemTypeCombined'] = df['Item'].apply(lambda x: x[0:2])
df['ItemTypeCombined'] = df['ItemTypeCombined'].map({'FD': 'Food','NC': 'Non-Consumable','DR': 'Drinks'})


# __Modify categories of Item_Fat_Content__

# In[22]:


#Mark non-consumables as separate category in low_fat:
df.loc[df['ItemTypeCombined'] == "Non-Consumable",'FatContent'] = "Non-Edible"
df['FatContent'].value_counts()


# __Check for presence of outliers__

# In[23]:


df.plot(kind='box',vert=False,figsize = (20,15))
plt.figure(figsize=(520,200))
plt.show()


# In[24]:


df.Visibility.plot(kind='box',vert=False,figsize = (20,5))
plt.figure(figsize=(20,5))
plt.show()


# __Treating the outliers in Visibility with maximum__

# In[25]:


Q1=df.Visibility.quantile(0.25)
Q3=df.Visibility.quantile(0.75)
IQR=Q3-Q1
Upper_Whisker = Q3+1.5*IQR


# In[26]:


df["Visibility"]= df["Visibility"].mask(df["Visibility"] >Upper_Whisker, Upper_Whisker)
df.Visibility.plot(kind='box',vert=False,figsize = (20,5))


# In[27]:


plt.figure(figsize=(8,5))
sns.countplot('FatContent',data= df,palette='ocean')


# __The Item bought of Low Fat.__

# In[28]:


plt.figure(figsize=(25,7))
sns.countplot('Itemtype',data= df,palette='summer')


# __Fruits and Vegetables are largely sold as people tend to use them on daily purpose__</br>
# __Snack Food are also sold a lot__</br>
# __Seafood are sold the least__

# In[29]:


plt.figure(figsize=(8,5))
sns.countplot('OutletSize',data= df,palette='spring')


# * __The outlets are more of small size__

# In[30]:


plt.figure(figsize=(8,5))
sns.countplot('OutletCity',data= df,palette='autumn')


# __The outlets are maximum in number in Tier 3 city__

# In[31]:


plt.figure(figsize=(8,5))
sns.countplot('OutletType',data= df,palette='flag')


# __The outlet are more of supermarket Type 1__

# In[32]:


plt.figure(figsize=(10,8))
sns.barplot(y='Itemtype', x='Sales',data= df,palette='autumn')


# __The products available were Fruits-Veggies and Snack Foods but the sales of Seafood and Starchy Foods seems higher__
# 
# __The sales can be improved with having stock of products that are most bought by customers.__

# In[33]:


df.columns


# In[34]:


plt.figure(figsize=(8,5))
plt.scatter('Visibility','Sales',data=df)
plt.xlabel('Sales')
plt.ylabel('Visibility')


# __Item_Visibility has a minimum value of zero.__</br>
# __This makes no practical sense because when a product is being sold in a store, the visibility cannot be 0.__
# 
# __Lets consider it like missing information and impute it with mean visibility of that product.__

# In[35]:


df['Visibility']= df['Visibility'].replace(0,df['Visibility'].mean())


# In[36]:


plt.figure(figsize=(8,5))
plt.scatter('Visibility','Sales',data=df)
plt.xlabel('Sales')
plt.ylabel('Visibility')


# __We can see that now visibility is not exactly zero and it has some value indicating that Item is rarely purchased by the customers.__

# In[37]:


plt.figure(figsize=(8,5))
plt.scatter(y='Sales',x='MRP',data=df)
plt.xlabel('Maximum Retail Price')
plt.ylabel('Sales')


# __Items with MRP ranging from 200-250 dollars is having high sales__

# In[38]:


plt.figure(figsize=(8,5))
sns.barplot(x='OutletCity',y='Sales',data= df,palette='plasma')


# __The Outlet Sales tend to be high for Tier3 and Tier 2 location types but we have only Tier3 locations maximum Outlets.__

# In[39]:


plt.figure(figsize=(8,5))
sns.lineplot(x='OutletYears',y='Sales',data= df,palette='viridis')


# __It is quiet evident that Outlets established 35 years before is having good Sales margin. <br>
# We also have a outlet which was established before 22 years has the lowest sales margin, so established years wouldn't improve the Sales unless the products are sold according to customer's interest.<br>
# Let us check the Vif for all features__

# In[40]:


df.columns


# In[41]:


plt.figure(figsize=(10,5))
sns.barplot('OutletCity','Sales',hue='OutletType',data= df,palette='magma')
plt.legend()


# __The Tier-3 location type has all types of Outlet type and has high sales margin.__

# __Numerical and One-Hot Coding of Categorical variables__

# In[42]:


#import libaries
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
var_mod = ['Item','FatContent','Itemtype','Outlet','OutletCity','OutletSize','ItemTypeCombined','OutletType']
for i in var_mod:
    df[i]=le.fit_transform(df[i])


# In[43]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.Series([variance_inflation_factor(df.values,idx)
              for idx in range(df.shape[1])],
             index=df.columns)
print(vif)


# In[44]:


df.drop('OutletEstablishmentYear',axis=1, inplace=True)
vif=pd.Series([variance_inflation_factor(df.values,idx)
              for idx in range(df.shape[1])],
             index=df.columns)
print(vif)


# In[45]:


df.drop('ItemTypeCombined',axis=1, inplace=True)
vif=pd.Series([variance_inflation_factor(df.values,idx)
              for idx in range(df.shape[1])],
             index=df.columns)
print(vif)


# In[46]:


df.drop('Outlet',axis=1, inplace=True)
vif=pd.Series([variance_inflation_factor(df.values,idx)
              for idx in range(df.shape[1])],
             index=df.columns)
print(vif)


# In[47]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),cmap='RdBu', annot=True)


# In[48]:


sns.pairplot(df)


# __As most of them are categorical column,we need to go for other statistical test__

# In[49]:


df.columns


# In[50]:


import statsmodels.formula.api as smf
import statsmodels.api as sm


# In[51]:


for col in df.drop(columns=['Item','Weight','MRP','Visibility','Sales']).columns:
    stats_model=smf.ols('Sales ~ ' + col, data=df).fit()
    print(sm.stats.anova_lm(stats_model,type=2))


# __We can see ['FatContent','Itemtype'] have little/no effect on th sales__

# In[52]:


df.drop(columns = ['FatContent','Itemtype'],axis=1,inplace=True)


# In[53]:


df= df.select_dtypes(exclude='object')


# In[54]:


df.columns


# In[55]:


#One Hot Coding:
df = pd.get_dummies(df, columns=['OutletCity','OutletSize','OutletType'])


# In[56]:


X= df.drop(columns = ['Sales'], axis=1)
y= df['Sales']


# In[57]:


X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=1)


# In[58]:


from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
x_train_minmax=mm_scaler.fit_transform(X_train)


# In[59]:


X_train=pd.DataFrame(x_train_minmax, columns=X.columns)


# In[60]:


x_test_minmax=mm_scaler.transform(X_valid)


# In[61]:


X_valid=pd.DataFrame(x_test_minmax, columns=X.columns)


# In[62]:


sns.kdeplot(df.Sales)


# In[63]:


sns.kdeplot(np.log(df.Sales))


# __let us also transform our target__

# In[64]:


y_train,y_valid = np.log(y_train),np.log(y_valid)


# ### Building Model

# # Linear Regression 

# In[65]:


df.columns


# In[66]:


LRmodel = LinearRegression(normalize=True)
LRmodel.fit(X_train,y_train)
y_pred = LRmodel.predict(X_valid)
print("Train Accuracy:",LRmodel.score(X_train,y_train))
print("Test Accuracy:",LRmodel.score(X_valid,y_valid))


# In[67]:


MSE=metrics.mean_squared_error(y_valid,y_pred)
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse)


# In[68]:


MSE=metrics.mean_squared_error(np.exp(y_valid),np.exp(y_pred))
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse)


# # Polynomial Regression

# In[73]:


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y_train)
training_data_prediction = lin_reg2.predict(X_poly)
print("Train Accuracy:",metrics.r2_score(y_train, training_data_prediction))
X_poly_test = poly.fit_transform(X_valid)
testing_data_prediction = lin_reg2.predict(X_poly_test)
print("Test Accuracy:",metrics.r2_score(y_valid, testing_data_prediction))
MSE= metrics.mean_squared_error(y_valid,testing_data_prediction)
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse)


# # Decision Tree

# In[75]:


dtregressor = DecisionTreeRegressor(random_state = 22)
dtregressor.fit(X_train,y_train)
print("Train Accuracy:",dtregressor.score(X_train,y_train))
print("Test Accuracy:",dtregressor.score(X_valid,y_valid))


# In[76]:


y_pred=dtregressor.predict(X_valid)
MSE= metrics.mean_squared_error(y_valid,y_pred)
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse)


# # XGB Regressor

# In[77]:


xgbregressor = XGBRegressor()
xgbregressor.fit(X_train, y_train)


# In[78]:


training_data_prediction = xgbregressor.predict(X_train)
print("Train Accuracy:",metrics.r2_score(y_train, training_data_prediction))
test_data_prediction = xgbregressor.predict(X_valid)
print("Test Accuracy:",metrics.r2_score(y_valid, test_data_prediction))
MSE= metrics.mean_squared_error(y_valid,test_data_prediction)
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse)


# # Random Forest Regression

# In[69]:


rfregressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
rfregressor.fit(X_train, y_train)
training_data_prediction = rfregressor.predict(X_train)
print("Train Accuracy:",metrics.r2_score(y_train, training_data_prediction))
test_data_prediction = rfregressor.predict(X_valid)
print("Test Accuracy:",metrics.r2_score(y_valid, test_data_prediction))
MSE= metrics.mean_squared_error(y_valid,test_data_prediction)
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse)


# # ADA Boost Regressor

# In[79]:


X, y = make_regression(n_features=4, n_informative=10,random_state=0, shuffle=False)
regr = AdaBoostRegressor(random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
training_data_prediction = regr.predict(X_train)
print("Train Accuracy:",metrics.r2_score(y_train, training_data_prediction))
test_data_prediction = regr.predict(X_valid)
print("Test Accuracy:",metrics.r2_score(y_valid, test_data_prediction))
MSE= metrics.mean_squared_error(y_valid,test_data_prediction)
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse)


# # Cross Validation

# In[86]:


def cross_val(model_name,model,X,y,cv):
    scores = cross_val_score(model, X, y, cv=cv)
    print(f'{model_name} Scores:')
    for i in scores:
        print(round(i,2))
    print(f'Average {model_name} score: {round(scores.mean(),2)}')


# In[87]:


from sklearn.model_selection import KFold
cv = KFold(n_splits=10, random_state=1, shuffle=True)
cross_val(LRmodel,LinearRegression(),X,y,cv)


# In[88]:


poly = PolynomialFeatures(degree=2)
cross_val(lin_reg2,LinearRegression(),poly.fit_transform(X),y,cv)


# In[82]:


cross_val(dtregressor,DecisionTreeRegressor(random_state = 22),X,y,cv)


# In[83]:


cross_val(xgbregressor,XGBRegressor(),X,y,cv)


# In[90]:


cross_val(rfregressor,RandomForestRegressor(n_estimators = 100, random_state = 0),X,y,10)


# In[89]:


cross_val(regr,AdaBoostRegressor(random_state=0, n_estimators=100),X,y,10)


# __Cross Validation on XGBRegressor and Random Forest Regressor yeilds a score of 0.84__

# In[ ]:




