#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


f_birth = pd.read_csv('daily-total-female-births-CA.csv', index_col=[0], parse_dates=[0])


# In[26]:


f_birth.head()


# In[27]:


type(f_birth)


# In[28]:


series_value = f_birth.values


# In[29]:


type(series_value)


# In[30]:


f_birth.size


# In[31]:


f_birth.tail()


# In[32]:


f_birth.describe()


# In[33]:


f_birth.plot()


# In[34]:


f_birth_mean = f_birth.rolling(window=20).mean()


# In[35]:


f_birth.plot()
f_birth_mean.plot()


# In[36]:


value = pd.DataFrame(series_value)


# In[44]:


birth_df = pd.concat([value,value.shift(1)], axis=1)


# In[45]:


birth_df.head()


# In[46]:


birth_df.columns = ['Actual_birth', 'Forecast_birth']


# In[47]:


birth_df.head()


# In[48]:


from sklearn.metrics import mean_squared_error
import numpy as np


# In[49]:


birth_test = birth_df[1:]


# In[50]:


birth_error = mean_squared_error(birth_test.Actual_birth, birth_test.Forecast_birth)


# In[51]:


birth_error


# In[52]:


np.sqrt(birth_error)


# In[53]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[54]:


plot_acf(f_birth)


# In[55]:


plot_pacf(f_birth)


# In[56]:


f_birth.size


# In[57]:


birth_train = f_birth[0:330]
birth_test = f_birth[330:365]


# In[58]:


birth_train.size


# In[59]:


birth_test.size


# In[60]:


from statsmodels.tsa.arima_model import ARIMA


# In[64]:


birth_model = ARIMA(birth_train, order=(2,1,3))


# In[65]:


birth_model_fit = birth_model.fit()


# In[66]:


birth_model_fit.aic


# In[67]:


birth_forecast = birth_model_fit.forecast(steps=35)[0]


# In[68]:


birth_forecast


# In[69]:


np.sqrt(mean_squared_error(birth_test, birth_forecast))


# In[ ]:




