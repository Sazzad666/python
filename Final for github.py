#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Basic Libraries
import numpy as np
import pandas as pd

#Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Text Handling Libraries
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


# In[52]:


df = pd.read_csv('D:/Python/BigBasket Products.csv',index_col='index')


# In[53]:


df.head()


# In[54]:


df.shape


# In[55]:


df.isnull().sum()


# In[56]:


print('Percentage Null Data In Each Column')
print('-'*30)
for col in df.columns:
    null_count = df[col].isnull().sum()
    total_count = df.shape[0]
    print("{} : {:.2f}".format(col,null_count/total_count * 100))


# In[57]:


print('Total Null Data')
null_count = df.isnull().sum().sum()
total_count = np.product(df.shape)
print("{:.2f}".format(null_count/total_count * 100))


# In[58]:


df = df.dropna()


# In[59]:


df.isnull().sum()


# In[60]:


df.shape


# In[62]:


# df.to_csv('data_cleaned.csv')


# In[63]:


df.head()


# In[64]:


counts = df['category'].value_counts()

counts_df = pd.DataFrame({'Category':counts.index,'Counts':counts.values})


# In[67]:


px.bar(data_frame=counts_df,
 x='Category',
 y='Counts',
 color='Counts',
 color_continuous_scale='blues',
 text_auto=True,
 title=f'Count of Items in Each Category')


# In[68]:


counts = df['sub_category'].value_counts()

counts_df_1 = pd.DataFrame({'Category':counts.index,'Counts':counts.values})[:10]


# In[69]:


px.bar(data_frame=counts_df_1,
 x='Category',
 y='Counts',
 color='Counts',
 color_continuous_scale='blues',
 text_auto=True,
 title=f'Top 10 Bought Sub_Categories')


# In[70]:


counts = df['brand'].value_counts()

counts_df_brand = pd.DataFrame({'Brand Name':counts.index,'Counts':counts.values})[:10]


# In[71]:


px.bar(data_frame=counts_df_brand,
 x='Brand Name',
 y='Counts',
 color='Counts',
 color_continuous_scale='blues',
 text_auto=True,
 title=f'Top 10 Brand Items based on Item Counts')


# In[72]:


counts = df['type'].value_counts()

counts_df_type = pd.DataFrame({'Type':counts.index,'Counts':counts.values})[:10]


# In[73]:


px.bar(data_frame=counts_df_type,
 x='Type',
 y='Counts',
 color='Counts',
 color_continuous_scale='blues',
 text_auto=True,
 title=f'Top 10 Types of Products based on Item Counts')


# In[74]:


def sort_recommendor(col='rating',sort_type = False):
    """
    A recommendor based on sorting products on the column passed.
    Arguments to be passed:
    
    col: The Feature to be used for recommendation.
    sort_type: True for Ascending Order
    """
    rated_recommend = df.copy()
    if rated_recommend[col].dtype == 'O':
        col='rating'
    rated_recommend = rated_recommend.sort_values(by=col,ascending = sort_type)
    return rated_recommend[['product','brand','sale_price','rating']].head(10)


# In[75]:


help(sort_recommendor)


# In[76]:


sort_recommendor(col='sale_price',sort_type=True)


# In[77]:


C= df['rating'].mean()
C


# In[78]:


def sort_recommendor(col='rating',sort_type = False):
    """
    A recommendor based on sorting products on the column passed.
    Arguments to be passed:
    
    col: The Feature to be used for recommendation.
    sort_type: True for Ascending Order
    """
    rated_recommend = df.copy().loc[df['rating'] >= 3.5]
    if rated_recommend[col].dtype == 'O':
        col='rating'
    rated_recommend = rated_recommend.sort_values(by=col,ascending = sort_type)
    return rated_recommend[['product','brand','sale_price','rating']].head(10)


# In[79]:


sort_recommendor(col='sale_price',sort_type=True)


# In[80]:


df.head()


# In[81]:


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
tfidf_matrix.shape


# In[82]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim


# In[83]:


indices = pd.Series(df.index, index=df['product']).drop_duplicates()

def get_recommendations_1(title, cosine_sim=cosine_sim):
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['product'].iloc[movie_indices]


# In[84]:


get_recommendations_1('Water Bottle - Orange')


# In[85]:


df2 = df.copy()


# In[86]:


df2.head()


# In[87]:


df2.shape


# In[88]:


rmv_spc = lambda a:a.strip()
get_list = lambda a:list(map(rmv_spc,re.split('& |, |\*|\n', a)))


# In[89]:


get_list('A & B, C')


# In[90]:


for col in ['category', 'sub_category', 'type']:
    df2[col] = df2[col].apply(get_list)


# In[91]:


df2.head()


# In[92]:


def cleaner(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[93]:


for col in ['category', 'sub_category', 'type','brand']:
    df2[col] = df2[col].apply(cleaner)


# In[94]:


df2.head()


# In[95]:


def couple(x):
    return ' '.join(x['category']) + ' ' + ' '.join(x['sub_category']) + ' '+x['brand']+' ' +' '.join( x['type'])
df2['soup'] = df2.apply(couple, axis=1)


# In[96]:


df2['soup'].head()


# In[97]:


df2.head()


# In[98]:


df2.to_csv('data_cleaned_1.csv')


# In[99]:


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])


# In[100]:


cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
cosine_sim2


# In[101]:


df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['product'])


# In[102]:


def get_recommendations_2(title, cosine_sim=cosine_sim):
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return df2['product'].iloc[movie_indices]


# In[103]:


old_rec = get_recommendations_1('Water Bottle - Orange').values
new_rec = get_recommendations_2('Water Bottle - Orange', cosine_sim2).values

pd.DataFrame({'Old Recommendor': old_rec,'New Recommendor':new_rec})


# In[104]:


old_rec = get_recommendations_1('Cadbury Perk - Chocolate Bar').values
new_rec = get_recommendations_2('Cadbury Perk - Chocolate Bar', cosine_sim2).values

pd.DataFrame({'Old Recommendor': old_rec,'New Recommendor':new_rec})


# In[ ]:




