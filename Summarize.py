import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#Removing Stop Word
def remove_stopwords(sen):
  sen_new = " ".join([i for i in sen if i not in stop_words])
  return sen_new
def summarize(sentences,amount=0.6):
    sentences=sent_tokenize(sentences)
    sentences=[x for x in sentences]
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    # Extract word vectors
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    sentence_vectors = []
    for i in clean_sentences:
      if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
      else:
        v = np.zeros((100,))
      sentence_vectors.append(v)
    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
      for j in range(len(sentences)):
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    # Graph to PageRank
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    NumberOfSentences=int(amount*len(sentences))
    s=[scores[x] for x in scores]
    s.sort(reverse=True)
    Threshold=s[NumberOfSentences]
    result=""
    for sen,key in zip(sentences,scores):
      if scores[key]>=Threshold:
        result=result+'\n\n'+sen
        
    return result
    
