
# coding: utf-8

# In[54]:


import os
import pandas as pd
arr = os.listdir('jsonfile')
type(arr)


# In[55]:


for filename in arr:
    print(filename)


# In[56]:


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


# In[57]:


len(dic)


# In[62]:


next(iter(dic.keys()))


# In[69]:


#titi={key:[value] for (key,value) in dic.items()}


# In[70]:


#toto=pd.DataFrame.from_dict(titi).transpose()


# In[71]:


#toto


# In[60]:


#next(iter(dic.values()))


# In[72]:


#1 create our corpus --> Combine it!
data_combined = {key: [value] for (key, value) in dic.items()}


# In[73]:


data_df=pd.DataFrame.from_dict(data_combined).transpose()


# In[74]:


data_df.columns=['transcript']


# In[75]:


data_df.index


# In[76]:


data_df


# In[77]:


#2-clean data


# In[78]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[79]:


def clean_text(doc):
    lema_text=[word.lemma_ for word in doc if word.is_alpha]
    return lema_text


# In[90]:


text='not at all that shik 09574040210122968 09revision 0agingwhen 0cindy 0when'
doc=nlp(text)
clean_text(doc)


# In[112]:


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


# In[113]:


data_cleaned=pd.DataFrame(data_df.transcript.apply(cleaned))


# In[114]:


data_cleaned.head()


# In[115]:


data_cleaned.index


# In[118]:


full_names = data_cleaned.index.tolist()


# In[119]:


full_names


# In[120]:


data_cleaned['artist_id']=full_names


# In[121]:


data_cleaned


# In[122]:


#data=data_cleaned.reset_index()


# In[123]:


# Let's pickle it for later use
data_cleaned.to_pickle("corpus.pkl")


# In[126]:


#data_cleaned.transcript[1]


# In[155]:


# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 

#stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)

stop_words = text.ENGLISH_STOP_WORDS.union(['pron','-PRON-','PRON'])
cv = CountVectorizer(stop_words=stop_words) 
dtm=cv.fit_transform(data_cleaned.transcript)
data_dtm = pd.DataFrame(dtm.toarray(), columns=cv.get_feature_names())


# In[156]:


dtm.toarray()


# In[157]:


cv.get_feature_names()


# In[158]:


data_dtm


# In[159]:


data_dtm.index=data_cleaned.index


# In[160]:


data_dtm


# In[161]:


# Let's pickle it for later use
data_dtm.to_pickle("dtm_artist.pkl")


# In[162]:


# Let's also pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object
import pickle
data_cleaned.to_pickle('data_clean.pkl')
pickle.dump(cv, open("cv.pkl", "wb"))

