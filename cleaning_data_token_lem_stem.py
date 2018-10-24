
# coding: utf-8

# In[177]:


import os
import pandas as pd
arr = os.listdir('jsonfile')
type(arr)


# In[178]:


for filename in arr:
    print(filename)


# In[179]:


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


# In[180]:


len(dic)


# In[181]:


next(iter(dic.keys()))


# In[182]:


next(iter(dic.values()))


# In[183]:


#for (key, value) in dic.items():
#    print(key)
#    print(value)
#    print(type(value))


# In[184]:


# Combine it!
data_combined = {key: [value] for (key, value) in dic.items()}


# In[185]:


import pandas as pd
pd.set_option('max_colwidth',150)

data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()
data_df


# In[198]:


import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])


# In[155]:


#text=[token.text.lower() for token in doc if token.is_alpha ]


# In[156]:


#text


# In[157]:


#nlp_practice = nlp(' '.join(text)) 


# In[158]:


#nlp_practice


# In[159]:


#practice = "practiced was great and accelerating" 
#nlp_practice = nlp(practice) 


# In[160]:


#[word.lemma_ for word in nlp_practice] 


# In[236]:


def clean_text(doc):
    #doc = nlp(text)
    lema_text=[word.lemma_ for word in doc if word.is_alpha]
    return lema_text


# In[247]:


import re
import string

def clean_textall(text):
    if len(text)>810000:
        text1 = text[:810000]
        text2=text[810000:]
        doc1 = nlp(text1)
        doc2=nlp(text2)
        cleantext1=clean_text(doc1)
        cleantext2=clean_text(doc2)
        lema_text=cleantext1+cleantext2
    else :
        doc=nlp(text)
        lema_text=clean_text(doc)
    return lema_text

cleaned = lambda x: clean_text(x)


# In[248]:


text=data_df.transcript.loc['Cindy Sherman']


# In[249]:


text


# In[250]:


len(text)


# In[251]:


toto=clean_textall(text)


# In[253]:


len(toto)


# In[228]:


doc1 = nlp('i dont know what to do')
doc2=nlp('i have a lot of eat that frog to do')

#text=[token.text.lower() for token in doc if token.is_alpha ]
#nlp_lema = nlp(' '.join(text)) 
#lema_text=[word.lemma_ for word in nlp_lema] 


# In[229]:


t1=clean_text(doc1)
t2=clean_text(doc2)


# In[230]:


t3=t1+t2


# In[231]:


t3


# In[223]:


lema_text=[word.lemma_ for word in doc if word.is_alpha]


# In[225]:


def clean_text(text):
    doc = nlp(text)
    lema_text=[word.lemma_ for word in doc if word.is_alpha]
    return lema_text


# In[224]:


lema_text


# In[191]:


clean=clean_text(text)


# In[192]:


clean


# In[199]:


data_cleaned=pd.DataFrame(data_df.transcript.apply(cleaned))


# In[101]:


data_cleaned.transcript


# In[102]:


data_cleaned.head()


# In[103]:


full_names=['Justin Mortimer','Pierre Buraglio']
data_cleaned['artist_name'] = full_names
data_cleaned


# In[104]:


data_cleaned.to_pickle("corpus_2_artist.pkl")


# In[105]:


# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_cleaned.transcript)


# In[106]:


cv.get_feature_names()


# In[107]:


data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_cleaned.index
data_dtm


# In[108]:


# Let's pickle it for later use
data_dtm.to_pickle("dtm_2_artist.pkl")


# In[ ]:


# Let's also pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object
data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open("cv.pkl", "wb"))
Additional Exercises

