
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:

get_ipython().magic('matplotlib inline')


# In[5]:

df = pd.read_csv("USA_Housing.csv")


# In[6]:

df.head(2)


# In[8]:

df.info() ## to get an idea on the type of information we have in the df


# In[9]:

df.describe() ## get an idea on statistics of our data set


# In[12]:

df.columns ## to return a list of the column names.


# In[15]:

sns.pairplot(df)


# In[19]:

sns.distplot(df["Price"], hist_kws=dict(edgecolor="k"))


# In[20]:

## Another thing we might want to do is generate a heatmap of the
## correlations between all the columns:



# In[21]:

df.corr()
# this will generate a correlation matrix between all the columns


# In[22]:

sns.heatmap(df.corr(), annot = True)


# In[23]:

## notice not a lost of the columns have correlations with eachother
## lets now use scikit-learn to train a linear regression model.


# In[24]:

## First thing we need to do is generate our 'x' and 'y' data:


# In[25]:

df.columns


# In[ ]:

## 'x' will be the features of the model


# In[33]:

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
X.head(1)


# In[27]:

## 'y' will be our target variable (what we are tryign to predict)


# In[29]:

y = df['Price']


# In[31]:

## we now have to split our data into a training set and a testing set.
## sklearn has a fucntion for splitting data into test and training sets.


# In[32]:

from sklearn.cross_validation import train_test_split


# In[34]:

## shift + tab to check documentation: train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .4, random_state = 101)
    ## test size = % of data that is used to train our model.
    ## random state is the set.seed for the random split.


# In[35]:

## now that we have our training and testing data we need to train our model
from sklearn.linear_model import LinearRegression


# In[36]:

## creation of linear regression object:
lm = LinearRegression()


# In[38]:

## we want to use .fit() to train our model
lm.fit(X_train, y_train)


# In[39]:

print(lm.intercept_)


# In[40]:

lm.coef_ 
## this will return the coeficients for each feature in our model


# In[41]:

## Each one of the featrues relates to a colum in our data frame
X_train.columns


# In[42]:

pd.DataFrame(lm.coef_,X.columns, columns = ['Coeff'])


# In[43]:

## from the chart above we can see that a one unit increase in the ave.Area income
## is associated with a 21.5$ increase in the price of the house.


# In[44]:

## Now we can practice getting predictions from our model


# In[45]:

predictions = lm.predict(X_test)


# In[47]:

predictions
## these are the predictions we get from our model.


# In[51]:

## We can test the accuracy of our model with the:
y_test.head(3)


# In[52]:

plt.scatter(y_test, predictions)


# In[53]:

## we can look at the risiduals of our testing data set
## remember residuals represent the difference between our model
## predictions and actual values.

sns.distplot((y_test-predictions))

## note if our data is not evenly distributed than we might need
## to revaluate if a lm is the correct model. 


# In[54]:

## Calling evaluation functions:


# In[55]:

from sklearn import metrics


# In[56]:

metrics.mean_absolute_error(y_test, predictions)


# In[57]:

metrics.mean_squared_error(y_test, predictions)


# In[58]:

np.sqrt(metrics.mean_squared_error(y_test, predictions))


# In[ ]:



