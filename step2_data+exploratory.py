
# coding: utf-8

# In[13]:


# Read in the document-term matrix
import pandas as pd

data = pd.read_pickle('dtm_artist.pkl')
data = data.transpose()
data.head()


# In[14]:


data['Cindy Sherman'].idxmax()


# In[16]:


data['Bridget Riley'].idxmax()


# In[17]:


# Find the top 30 words said by each comedian
top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

top_dict


# In[20]:


# Print the top 15 words for each artist
for artist, top_words in top_dict.items():
    print(artist)
    print(', '.join([word for word, count in top_words[0:20]]))
    print('---')


# In[23]:


# Look at the most common top words --> add them to the stop word list
from collections import Counter

# Let's first pull out the top 30 words for each artist
words = []
for artist in data.columns:
    top = [word for (word, count) in top_dict[artist]]
    for t in top:
        words.append(t)
        
words


# In[24]:


# Let's aggregate this list and identify the most common words along with how many routines they occur in
Counter(words).most_common()


# In[100]:


# If more than half of the comedians have it as a top word, exclude it from the list
add_stop_words = [word for word, count in Counter(words).most_common() if count > 1]
add_stop_words=['PRON','pron','say','like','cindy', 'sherman','gerhard', 'richter','bridget', 'riley','new','york','twitter','facebook','pinterest','annual']


# In[101]:


add_stop_words


# In[102]:


# Let's update our document-term matrix with the new list of stop words
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer

# Read in cleaned data
data_clean = pd.read_pickle('data_clean.pkl')

# Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate document-term matrix
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.transcript)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

# Pickle it for later use
import pickle
pickle.dump(cv, open("cv_stop.pkl", "wb"))
data_stop.to_pickle("dtm_stop.pkl")


# In[103]:


# Let's make some word clouds!
# Terminal / Anaconda Prompt: conda install -c conda-forge wordcloud
from wordcloud import WordCloud

wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)


# In[104]:


wc


# In[105]:


# Reset the output dimensions
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 6]

full_names = ['Cindy Sherman','Gerhard Richter','Bridget Riley']

# Create subplots for each comedian
for index, artist in enumerate(data.columns):
    wc.generate(data_clean.transcript[artist])
    
    plt.subplot(3, 4, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(full_names[index])
    
plt.show()


# In[106]:


# Find the number of unique words that for each artist
# Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once
unique_list = []
for comedian in data.columns:
    uniques = data[comedian].nonzero()[0].size
    unique_list.append(uniques)

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['artist', 'unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')
data_unique_sort

