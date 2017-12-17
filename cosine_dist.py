from scipy.spatial.distance import cosine
import numpy as np
import tensorflow as tf
import spacy

nlp = spacy.load('fr')

doc1 = nlp('Bonjour')
doc2 = nlp('Bonjour comment ça va')

doc2_1 = nlp('comment')
doc2_2 = nlp('ça')
doc2_3 = nlp('va')
doc3 = nlp('kikou lol')

ref = doc1.vector
input1 = doc1.vector + 0.1 * doc2_1.vector + 0.01 * doc2_2.vector + 0.01 * doc2_3.vector
input2 = (doc1.vector + 0.1 * doc2_1.vector + 0.1 * doc2_2.vector + 0.1 * doc2_3.vector) / 1.3
input3 = doc2.vector
input4 = doc3.vector

#x = tf.constant(input1)
#y = tf.constant(input2)
#z = tf.constant(input3)
#with tf.device('/device:CPU:0'):
sess = tf.Session()
"""
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
dist = (1 - tf.losses.cosine_distance(tf.nn.l2_normalize(x, 0), tf.nn.l2_normalize(y, 0), dim=0))

res = [sess.run(dist, feed_dict={x:ref, y:input_val}) for input_val in [input1, input2, input3, input4]]
print(res)
"""
#b = tf.placeholder(tf.float32, shape=[None, 384], name="input_placeholder_b")
#with tf.device('/device:GPU:0'):
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
normalize_a = tf.nn.l2_normalize(a, dim=1)
normalize_b = tf.nn.l2_normalize(b, dim=1)
#cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
cos_similarity = tf.matmul(normalize_a, tf.transpose(normalize_b, [1, 0]))
#cos_similarity = tf.argmax(cos_similarity, 1)
cos_sim=sess.run(cos_similarity,feed_dict={a:[ref],b:[input1,input2,input3,input4]})
print("result: %s" % cos_sim)

print(np.argmax(cos_sim))