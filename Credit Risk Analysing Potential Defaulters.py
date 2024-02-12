#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.pyplot import figure
warnings.filterwarnings(action="ignore")
pd.set_option("display.max_columns",500)
pd.set_option("display.max_rows",500)


# In[4]:


df=pd.read_csv("application_data.csv")
df.head()


# In[4]:


df.shape


# In[6]:


df.info(verbose=True, show_counts=True)


# In[7]:


df.dtypes


# ## Converting days_birth to age

# In[8]:


df["AGE"]=df["DAYS_BIRTH"]/(-365)


# In[11]:


## creating age groups
## create the bucket <30,30-40,40-50,50-60,60+
df["AGE_GROUP"]=pd.cut(df.AGE,bins=[0,30,40,50,60,9999],labels=["<30","30-40","40-50","50-60","60+"])


# ## Converting Days Employed to Years

# In[12]:


df["DAYS_EMPLOYED"].value_counts()


# In[13]:


df["DAYS_EMPLOYED"]=df["DAYS_EMPLOYED"].replace(365243,np.NaN)


# In[14]:


df["DAYS_EMPLOYED"].value_counts()


# In[15]:


## Creating years bucket
df["YEARS_EMPLOYED_GRP"]=pd.cut(df["DAYS_EMPLOYED"],bins=[0,5,10,15,20,9999],labels=["0-5","5-10","10-15","15-20","20+"])


# In[16]:


## Converting days registration to years
df["DAYS_REGISTRATION"].value_counts()


# In[17]:


df["REGISTRATION_YEARS"]=df["DAYS_REGISTRATION"]/(-365)


# In[18]:


df["REGISTRATION_YEARS"]


# ## Converting days_id_publish to years

# In[20]:


df["YEARS_ID_PUBLISH"]=df["DAYS_ID_PUBLISH"]/(-365)
# plotting the years_id_publish column
sns.boxplot(df["YEARS_ID_PUBLISH"])
plt.title("Box Plot for YEARS_ID_PUBLISH")
plt.show()


# ## Splitting Amt_income_total into buckets for easy analysis

# In[21]:


df["AMT_INCOME_GROUP"]=pd.cut(df["AMT_INCOME_TOTAL"],bins=[0,100000,200000,300000,400000,500000],labels=["V.LOW","LOW","MEDIUM","HIGH","V.HIGH"])


# In[23]:


df["BLN_OWN_CAR"]=df["FLAG_OWN_CAR"].apply(lambda x:1 if(x =="Y") else 0)


# In[24]:


df["BLN_OWN_REALTY"]=df["FLAG_OWN_REALTY"].apply(lambda x:1 if(x =="Y") else 0)


# ## Handling missing data

# In[25]:


#Analysing organization type column
df["ORGANIZATION_TYPE"].value_counts()


# ## Large number of records with value XNA could denote missing values. Hence replacing it with NaN

# In[26]:


df["ORGANIZATION_TYPE"]= df["ORGANIZATION_TYPE"].replace("XNA",np.NaN)


# In[30]:


df.isnull().sum()/len(df)*100


# ## Keeping a threshold of 40% missing data to remove columns

# In[34]:


df = df[df.columns[df.isnull().mean()<.4]]


# In[35]:


df.head(20)


# In[36]:


df.shape


# In[37]:


#get info about existing columns
df.info(verbose=True, show_counts=True)


# ### Checking Outliers.
# ### Outliers are data which are different and do not fall into the normal distribution of data. One common visualization used to detect outliers is box plot.

# In[38]:


sns.boxplot(df["CNT_CHILDREN"])
plt.title("Box Plot for children count")
plt.show()


# In[39]:


# checking for outliers in amount credit
print(df["AMT_CREDIT"].quantile([0.0,0.25,0.5,0.75,0.90,0.95,0.99,1.0]))
sns.boxplot(df["AMT_CREDIT"])
plt.title("Box Plot for Amount Credit")
plt.show()


# In[41]:


df.columns


# In[42]:


df[['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE',
       'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
       'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL',
       'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
       'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS',
       'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
       'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
       'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
       'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
       'ORGANIZATION_TYPE', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
       'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'AGE',
       'AGE_GROUP', 'REGISTRATION_YEARS', 'YEARS_ID_PUBLISH',
       'AMT_INCOME_GROUP', 'BLN_OWN_CAR', 'BLN_OWN_REALTY']].describe(percentiles=[.05,.25,.5,.75,.95])


# In[44]:


plt.figure(figsize = (15, 10))
plt.subplot(3, 2, 1)
sns.boxplot(x ='CNT_CHILDREN', data = df)
plt.subplot(3, 2, 2)
sns.boxplot(x ='AMT_INCOME_TOTAL', data = df)
plt.subplot(3, 2, 3)
sns.boxplot(x ='AMT_ANNUITY', data = df)
plt.subplot(3, 2, 4)
sns.boxplot(x ='AMT_CREDIT', data = df)
plt.subplot(3, 2, 5)
sns.boxplot(x ='REGISTRATION_YEARS', data = df)
plt.show()


# In[45]:


df["AMT_INCOME_TOTAL"].describe()


# In[46]:


#Creating bins to convert AMT_INCOME_TOTAL into the categorical values
df["INCOME_BRACKET"]=pd.cut(df["AMT_INCOME_TOTAL"],[0,100000,125000,175000,225000,1000000000],labels=["Very Low", "Low", "Medium", "High", "Very High"])


# In[47]:


df["INCOME_BRACKET"]


# In[48]:


df['AMT_CREDIT'].describe()


# In[49]:


##Creating bins to convert 'AMT_CREDIT' into categorical value

df['CREDIT_BRACKETS']=pd.cut(df['AMT_CREDIT'],[0,200000,500000,800000,1000000,1000000000],labels=['Very Low','Low','Medium','High','Very High'])


# In[50]:


df["CREDIT_BRACKETS"]


# ### EXPLORATORY DATA ANALYSIS (EDA)

# In[51]:


print('Percentage of people with payment difficulties : ',100*round(len(df[df.TARGET==1])/len(df),4),'%')
print('Percentage of people with no payment diffiiculties : ',100*round(len(df[df.TARGET==0])/len(df),4),'%')


# ## Organization Type

# In[56]:


plt.figure(figsize=(15,10))
plt.title("Bar Chart That Shows The Distribution Of Organization Type")
df["ORGANIZATION_TYPE"].value_counts(normalize=True).sort_values(ascending=True).plot.barh()
plt.xticks(rotation=90)
plt.show()


# ## Percentage of defualters in each Organization Type

# In[58]:


plt.figure(figsize=(15,10))
plt.title("Percentage Of Defaulters In Each Organization Type")
df.groupby("ORGANIZATION_TYPE")["TARGET"].mean().sort_values(ascending=True).plot.barh()
plt.show()


# ## Income Group Distribustion

# In[61]:


plt.figure(figsize=(15,10))
plt.title("Income Group Distribution")
df["INCOME_BRACKET"].value_counts(normalize=True).plot.bar()
plt.xticks(rotation=90)
plt.show()


# ## Percentage of Defaulters by Income Bracket

# In[64]:


plt.figure(figsize=(15,10))
plt.title("Percentage Of Defaulters By Income Bracket")
df.groupby("INCOME_BRACKET")["TARGET"].mean().sort_values(ascending=False).plot.bar()
plt.show()


# In[71]:


df["CODE_GENDER"].value_counts()


# In[72]:


df["CODE_GENDER"]= df["CODE_GENDER"].replace("XNA",np.NaN)


# ## Gender Distribution

# In[73]:


plt.figure(figsize=(15,10))
plt.title("Gender Distribution")
df["CODE_GENDER"].value_counts(normalize=True).plot.bar()
plt.xticks(rotation=90)
plt.show()


# ## Percentage Of Defaulters By Gender

# In[75]:


plt.figure(figsize=(15,10))
plt.title("The Percentage Of Defaulters By Gender")
df.groupby("CODE_GENDER")["TARGET"].mean().sort_values(ascending=False).plot.bar()
plt.show()


# ## Age Distribution

# In[76]:


plt.figure(figsize=(15,10))
plt.title("Age Distribution")
df["AGE_GROUP"].value_counts(normalize=True).plot.bar()
plt.xticks(rotation=90)
plt.show()


# ## Percentage Of Defaulters By Age Group 

# In[77]:


plt.figure(figsize=(15,10))
plt.title("The Percentage Of Defaulters By Age Group")
df.groupby("AGE_GROUP")["TARGET"].mean().sort_values(ascending=False).plot.bar()
plt.show()


# ## Type Of Loan Distribution

# In[80]:


plt.figure(figsize=(15,10))
plt.title("Contract Type Distribution")
df["NAME_CONTRACT_TYPE"].value_counts(normalize=True).plot.bar()
plt.xticks(rotation=90)
plt.show()


# ## Distribution Of Occupation Type

# In[81]:


plt.figure(figsize=(15,10))
plt.title("Distribution Of Occupation Type")
df["OCCUPATION_TYPE"].value_counts(normalize=True).plot.bar()
plt.xticks(rotation=90)
plt.show()


# ## The Percentage Of Defaulters By Occupation Type

# In[82]:


plt.figure(figsize=(15,10))
plt.title("The Percentage Of Defaulters By Occupation Type")
df.groupby("OCCUPATION_TYPE")["TARGET"].mean().sort_values(ascending=False).plot.bar()
plt.show()


# ## Distribution Of Education Type 

# In[83]:


plt.figure(figsize=(15,10))
plt.title("Distribution Of Education Type")
df["NAME_EDUCATION_TYPE"].value_counts(normalize=True).plot.bar()
plt.xticks(rotation=90)
plt.show()


# In[84]:


plt.figure(figsize=(15,10))
plt.title("The Percentage Of Defaulters By Education Type")
df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values(ascending=False).plot.bar()
plt.show()


# In[88]:


df["NAME_FAMILY_STATUS"].value_counts()


# In[89]:


df["NAME_FAMILY_STATUS"]= df["NAME_FAMILY_STATUS"].replace("unknown",np.NaN)


# ### Distribution Of The Family Status

# In[19]:


plt.figure(figsize=(15,10))
plt.title("Distribution Of Family Status")
df["NAME_FAMILY_STATUS"].value_counts(normalize=True).plot.bar()
plt.xticks(rotation=90)
plt.show()


# ## Percentage Of Defualters By Family Status

# In[86]:


plt.figure(figsize=(15,10))
plt.title("The Percentage Of Defaulters By Family Status")
df.groupby("NAME_FAMILY_STATUS")["TARGET"].mean().sort_values(ascending=False).plot.bar()
plt.show()


# In[92]:


#Analysing house type 
#Plot a pie chart
plt.figure(figsize=(13,8))
plt.title("Pie chart that shows distribution of house type")
df["NAME_HOUSING_TYPE"].value_counts().plot.pie()
plt.show()


# In[93]:


df["NAME_INCOME_TYPE"].value_counts()


# ## Distribution Of Income Type

# In[94]:


plt.figure(figsize=(15,10))
plt.title("Distribution Of Income Type")
df["NAME_INCOME_TYPE"].value_counts(normalize=True).plot.bar()
plt.xticks(rotation=90)
plt.show()


# ## Percentage Of Defaulters By Income Type

# In[96]:


plt.figure(figsize=(15,10))
plt.title("The Percentage Of Defaulters By Income Type")
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean().sort_values(ascending=False).plot.bar()
plt.show()


# ### Distribution Of Credit Bracket 

# In[98]:


plt.figure(figsize=(15,10))
plt.title("Distribution Of Credit Bracket")
df["CREDIT_BRACKETS"].value_counts(normalize=True).plot.bar()
plt.xticks(rotation=90)
plt.show()


# ## Percentage Of Defaulters By Credit Brackets

# In[99]:


plt.figure(figsize=(15,10))
plt.title("The Percentage Of Defaulters By Credit Brackets")
df.groupby("CREDIT_BRACKETS")["TARGET"].mean().sort_values(ascending=False).plot.bar()
plt.show()


# ## Load Previous Application Data

# In[6]:


df_prevapp = pd.read_csv('previous_application.csv')
df_prevapp.head()


# In[101]:


df_prevapp.shape


# In[102]:


df_prevapp.dtypes


# In[103]:


df_prevapp.isna().sum()


# In[7]:


# Merge previous application data with current data.
df_merge = pd.merge(left=df,right=df_prevapp, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR')
df.head(20)


# In[14]:


#Gender-wise breakdown of the previous loan application status across target values

plt.figure(figsize=(15,10))
plt.subplot(1, 2, 1)
plt.title('TARGET = 0')
sns.countplot(x='NAME_CONTRACT_STATUS',hue='CODE_GENDER',data=df_merge)
plt.subplot(1, 2, 2)
plt.title('TARGET = 1')
sns.countplot(x='NAME_CONTRACT_STATUS',hue='CODE_GENDER',data=df_merge)
plt.show()


# ### CONCLUSION

# ## The following Groups are more likely to default.

# ## 1. Low income group
# ## 2. Age group <30
# ## 3. Low Skilled Laborers occupation type.
# ## 4. Transport Type 3 organization Type.
# ## 5. Lower Secondary Education type.

# In[ ]:




