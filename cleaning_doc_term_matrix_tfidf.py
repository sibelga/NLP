
# coding: utf-8

# In[1]:


import os
import pandas as pd
arr = os.listdir('jsonfile')
type(arr)


# In[2]:


for filename in arr:
    print(filename)


# In[3]:


dic={}
for filename in arr:
    if '.DS_Store' not in filename:
        print(filename)
        file= pd.read_json('jsonfile/'+filename)
        artist=str(file['artistName'][0])
        print(artist)
        text= ' '.join(file['text'])
        dic[artist]=text
#print(dic)


# In[4]:


len(dic)


# In[5]:


next(iter(dic.keys()))


# In[7]:


#1 create our corpus --> Combine it!
data_combined = {key: [value] for (key, value) in dic.items()}


# In[8]:


data_df=pd.DataFrame.from_dict(data_combined).transpose()


# In[9]:


data_df.columns=['transcript']


# In[10]:


data_df.index


# In[11]:


data_df


# In[12]:


#2-clean data


# In[13]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[14]:


def clean_text(doc):
    lema_text=[word.lemma_ for word in doc if word.is_alpha]
    return lema_text


# In[15]:


#text='not at all that shik 09574040210122968 09revision 0agingwhen 0cindy 0when'
#doc=nlp(text)
#clean_text(doc)


# In[16]:


def clean_textall(text):
    #print(len(text))
    if len(text)>990000:
        #print('its big')
        text1 = text[:990000]
        text2=text[990000:]
        doc1 = nlp(text1)
        doc2=nlp(text2)
        cleantext1=clean_text(doc1)
        cleantext2=clean_text(doc2)
        lema_text=cleantext1+cleantext2
        lema_text=' '.join(lema_text)
    else :
        #print('not big')
        doc=nlp(text)
        lema_text=clean_text(doc)
        lema_text=' '.join(lema_text)
    return lema_text

cleaned = lambda x: clean_textall(x)


# In[17]:


data_cleaned=pd.DataFrame(data_df.transcript.apply(cleaned))


# In[18]:


data_cleaned.head()


# In[19]:


data_cleaned.index


# In[20]:


full_names = data_cleaned.index.tolist()


# In[21]:


full_names


# In[22]:


data_cleaned['artist_id']=full_names


# In[23]:


data_cleaned


# In[24]:


#data=data_cleaned.reset_index()


# In[27]:


# Let's pickle it for later use
data_cleaned.to_pickle("corpus.pkl")


# In[26]:


#data_cleaned.transcript[1]


# In[34]:


# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 

#stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)

stop_words = text.ENGLISH_STOP_WORDS.union(['pron','-PRON-','PRON'])
tfidf_vectorizer = TfidfVectorizer(max_df=0.8,
                                 min_df=0.2, stop_words=stop_words,
                                 use_idf=True)



dtm=tfidf_vectorizer.fit_transform(data_cleaned.transcript)
data_dtm = pd.DataFrame(dtm.toarray(), columns=tfidf_vectorizer.get_feature_names())


# In[35]:


dtm.toarray()


# In[36]:


tfidf_vectorizer.get_feature_names()


# In[37]:


data_dtm


# In[38]:


data_dtm.index=data_cleaned.index


# In[39]:


data_dtm


# In[40]:


# Let's pickle it for later use
data_dtm.to_pickle("dtm_tfidf.pkl")


# In[41]:


# Let's also pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object
import pickle
data_cleaned.to_pickle('data_clean.pkl')
pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pkl", "wb"))

