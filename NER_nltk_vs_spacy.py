
# coding: utf-8

# In[38]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import spacy
nlp = spacy.load('en_core_web_sm')
#nltk.download()


# In[39]:


ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'


# In[40]:


#Then we apply word tokenization and part-of-speech tagging to the sentence.
def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


# In[41]:


preprocess(ex)


# In[42]:


doc=nlp(ex)
[(token.text,token.pos_,token.tag_) for token in doc]


# In[11]:


import spacy
# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')


# In[13]:


from spacy.tokenizer import Tokenizer
tokenizer = Tokenizer(nlp.vocab)


# In[45]:


toto=tokenizer(ex)


# In[50]:


[t for t in toto if t.is_alpha]

