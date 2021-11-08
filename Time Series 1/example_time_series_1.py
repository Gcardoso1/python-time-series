#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


shampoo = pd.read_csv(r'shampoo_sales.csv')


# In[12]:


shampoo.head()


# In[13]:


type(shampoo)


# In[14]:


shampoo = pd.read_csv(r'shampoo_sales.csv', index_col=[0], parse_dates=True, squeeze=True)


# In[15]:


type(shampoo)


# In[16]:


shampoo.plot()


# In[17]:


shampoo.plot(style='k.')


# In[18]:


shampoo.size


# In[19]:


shampoo.describe()


# In[20]:


shampoo_ma = shampoo.rolling(window=10).mean()


# In[21]:


shampoo_ma.plot()


# In[23]:


shampoo


# In[24]:


shampoo_base = pd.concat([shampoo, shampoo.shift(1)],axis=1)


# In[25]:


shampoo_base


# In[26]:


shampoo_base.columns = ['Actual_Sales', 'Forecast_Sales']


# In[27]:


shampoo_base.head()


# In[28]:


shampoo_base.dropna(inplace=True)


# In[29]:


shampoo_base


# In[30]:


shampoo_base.head()


# In[31]:


from sklearn.metrics import mean_squared_error
import numpy as np


# In[32]:


shampoo_error = mean_squared_error(shampoo_base.Actual_Sales, shampoo_base.Forecast_Sales)


# In[33]:


shampoo_error


# In[34]:


np.sqrt(shampoo_error)


# In[35]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[36]:


plot_acf(shampoo)


# In[37]:


plot_pacf(shampoo)


# In[38]:


from statsmodels.tsa.arima_model import ARIMA


# In[39]:


shampoo_train = shampoo[0:25]
shampoo_test = shampoo[25:36]


# In[46]:


shampoo_model = ARIMA(shampoo_train,order=(3,2,3))


# In[47]:


shampoo_model_fit = shampoo_model.fit()


# In[48]:


shampoo_model_fit.aic


# In[49]:


shampoo_forecast = shampoo_model_fit.forecast(steps=11)[0]


# In[50]:


np.sqrt(mean_squared_error(shampoo_test, shampoo_forecast))


# In[53]:


p_values = range(0,5)
d_values = range(0,3)
q_values = range(0,5)


# In[54]:


import warnings
warnings.filterwarnings("ignore")


# In[56]:


for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p,d,q)
            train,test = shampoo[0:25], shampoo[25:36]
            predictions = list()
            for i in range(len(test)):
                try:
                    model = ARIMA(train, order)
                    model_fit = model.fit(disp=0)
                    pred_y = model_fit.forecast()[0]
                    predictions.append(pred_y)
                    error = mean_squared_error(test,predictions)
                    print('ARIMA%s MSE = %.2f'% (order,error))
                except:
                    continue


# In[ ]:




