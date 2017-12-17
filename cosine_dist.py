from scipy.spatial.distance import cosine
import numpy as np
import tensorflow as tf
import spacy
#from spacy.tokenizer import Tokenizer
from spacy.lang.fr import French
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load('fr')
#tokenizer = Tokenizer(nlp.vocab)
tokenizer = French().Defaults.create_tokenizer(nlp)
def word_tokenizer(text):
    return [ token.text for token in tokenizer(text)]


text1 = 'bonjour, qui êtes vous ?'
text2 = "bonjour comment ça va, qu'est-ce que tu deviens ?"
text3 = "qui est le président des états unis ?"
text4 = "qui est la bas ?"

print(word_tokenizer(text2))
tfidf = TfidfVectorizer(tokenizer=tokenizer, use_idf=True)
sparse_tfidf_texts = tfidf.fit_transform([text1, text2, text3, text4])
print(sparse_tfidf_texts)
#with tf.device('/device:CPU:0'):

"""
sess = tf.Session()
#b = tf.placeholder(tf.float32, shape=[None, 384], name="input_placeholder_b")
#with tf.device('/device:GPU:0'):
a = tf.placeholder(tf.float32, shape=[None, 384], name="input_placeholder_a")
b = tf.placeholder(tf.float32, shape=[None, 384], name="input_placeholder_b")
normalize_a = tf.nn.l2_normalize(a, dim=1)
normalize_b = tf.nn.l2_normalize(b, dim=1)
#cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
cos_similarity = tf.matmul(normalize_a, tf.transpose(normalize_b, [1, 0]))
#cos_similarity = tf.argmax(cos_similarity, 1)

cos_sim=sess.run(cos_similarity,feed_dict={a:[ref],b:[input1,input2,input3,input4]})
print("result: %s" % cos_sim)

print(np.argmax(cos_sim))
"""