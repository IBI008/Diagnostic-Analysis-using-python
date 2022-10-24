#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the Library.
import pandas as pd

ad = pd.read_csv('actual_duration.csv')
ar = pd.read_csv('appointments_regional.csv')
nc = pd.read_excel('national_categories.xlsx')

# View the ad DataFrame.
print(ad.shape)
print(ad.dtypes)
print(ad.head())
print(ad.tail())
ad.isna().sum()


# In[2]:


# View the ar DataFrame.
print(ar.shape)
print(ar.dtypes)
print(ar.head())
print(ar.tail())
ar.isna().sum()


# In[3]:


# View the nc DataFrame.
print(nc.shape)
print(nc.dtypes)
print(nc.head())
print(nc.tail())
nc.isna().sum()


# In[4]:


# Determine the descriptive analysis.
# Use the describe function.
print(ad.describe())


# In[5]:


print(ar.describe())


# In[6]:


print(nc.describe())


# In[7]:


# Question one
# How many location are there in the dataset.
print(f"The number of location in the dataset is {ad ['sub_icb_location_name'].count()}")


# In[8]:


# Question two:
# What are the five location with highest number of records
ad['sub_icb_location_name'].value_counts()


# In[9]:


# Question three:
# How many service settings, context types, national categories,
# and appointment status are there
nc['service_setting'].value_counts()


# In[10]:


nc['context_type'].value_counts()


# In[11]:


nc['national_category'].value_counts()


# In[12]:


ar['appointment_status'].value_counts()


# In[13]:


nc.shape


# In[14]:


ar.shape


# In[15]:


ad.shape


# In[16]:


# Assignment Activity three
# Analyse Data


# In[17]:


# Question one 
# Between what date were appointments scheduled?
ad['appointment_date'].min()


# In[18]:


ad['appointment_date'].max()


# In[19]:


nc['appointment_date'].min()


# In[20]:


# Question Two.
# Which service setting reported the most appointments in North
# West London from 1 January to 1 June 2022?
ad_subset = pd.read_csv('actual_duration.csv',
                       usecols=['sub_icb_location_code'])
ad_subset.head()


# In[21]:


# Question three
# which month has the highest number of appointment.
nc_m = pd.


# In[22]:


ad.columns


# In[25]:


import pandas as pd
nc = pd.read_excel('national_categories.xlsx')
nc.shape


# In[26]:


# Assignment Activity four.


# In[30]:


# Import Matplotlib, Seaborn, NumPy, and Pandas.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime


# In[ ]:


# Set the figure size.
sns.set(rc={'figure.figsize':(15, 12)})

# Set the tick style.
sns.set_style('ticks')

# Set the colour style.
sns.set_style('white')


# In[28]:


# Create three visualisations indicating the number of appointments per month for service settings, 
# context types, and national categories?


# In[29]:


nc_ss = nc .groupby('appointment_month')[['service_setting']] .sum() .reset_index() .copy()


# In[31]:


print(nc_ss.head())


# In[32]:


sns.lineplot(x='appointment_month', y='service_setting', data=nc_ss, ci=None)


# In[33]:


nc_ct = nc .groupby('appointment_month')[['context_type']] .sum() .reset_index() .copy()


# In[34]:


print(nc_ct.head())


# In[35]:


sns.lineplot(x='appointment_month', y='context_type', data=nc_ct, ci=None)


# In[ ]:


nc_nc = nc .groupby('appointment_month')[['national_category']] .sum() .reset_index() .copy()


# In[ ]:


print(nc_nc.head())


# In[ ]:


sns.lineplot(x='appointment_month', y='national_category', data=nc_nc, ci=None)


# In[ ]:


# Question two.Create four visualisations indicating the number of appointments for 
# service setting per season. The seasons are summer (August 2021), autumn (October 2021), 
# winter (January 2022), and spring (April 2022)?


# In[36]:


nc_ss_d = nc.groupby(['appointment_date', 'appointment_month', 'service_setting'])
nc_ss_d


# In[37]:


nc_ss_d.sum()


# In[38]:


sns.lineplot(x='appointment_month', y='service_setting', data=nc_ss_d)


# In[39]:


# Question two continuation


# In[40]:


# Assignment Activity five


# In[41]:


# Import Seaborn and Pandas.
import seaborn as sns
import pandas as pd


# In[42]:


# Set the figure size.
sns.set(rc={'figure.figsize':(15, 12)})

# Set the tick style.
sns.set_style('ticks')

# Set the colour style.
sns.set_style('white')


# In[43]:


pd.options.display.max_colwidth=200


# In[44]:


# Read csv file from the current directory.
tweets = pd.read_csv('tweets.csv')

# View the Dataframe
print(tweets.columns)
print(tweets.shape)


# In[45]:


tweets.head()


# In[46]:


tweets.describe()


# In[47]:


tweets.info()


# In[48]:


tweets['tweet_retweet_count'].value_counts()


# In[49]:


tweets['tweet_favorite_count'].value_counts()


# In[50]:


# Create a new DataFrame.
tweets_txt_pd = pd.DataFrame(['tweet_full_text'])

# View the DataFrame.
tweets_txt_pd


# In[51]:


tags = []

for y in [x.split(' ') for x in tweets['tweet_full_text'].values]:
    for z in y:
        if '#' in z:
            # Change to lowercase.
            tags.append(z.lower())
            
# Create output
tags


# In[52]:


# Convert extracted data into a pandas dataframe
# Create a DataFrame directly from the output.
data = pd.DataFrame(tags)

# View the DataFrame.
data.head()


# In[53]:


# Convert, clean and extract.

# Save the DataFrame as a CSV file with index.
data.to_csv('ref.csv', index=False)


# In[ ]:




