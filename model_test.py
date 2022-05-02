# from keras.engine import sequential
import numpy as np
from string import punctuation
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as l
import gensim
import random
from sklearn.model_selection import train_test_split

data = np.array(['Buy the stock.', 'Sell stock now.', 'I dont know how to feel about stock.', 'I like tesla.',
                 'Tesla is really bad.', 'Help me im going insane.', 'what a nice day.', 'i had pizza for breakfast.',
                 'Im an adult.', 'hello world.']).T
context = np.array(
    [[2, 3, 0, -1, 0], [-4, -2, 1, 0, 1], [1, 0, -1, 0, 1], [-1, -1, -2, -1, -2], [1, 1, 2, 1, 2], [3, 4, 0, -1, -3],
     [1, -1, 1, -1, 1], [-1, 1, -1, 1, -1], [2, 3, -4, -2, 1], [0, 0, 0, 0, 0]])
my_labels = np.array([5, -8, 0, 2, -2, -5, 1, -1, 3, 0]).T

'''
features_csv = pd.read_csv('features.csv')
labels_csv = pd.read_csv('lables.csv') # not labels
my_labels = np.array(labels_csv['Close'].tolist())

tweets = features_csv['text'].tolist()
# print(tweets[0])
# for s in range(len(tweets)):
#     tweets[s] = tweets[s] + 'bunga'
#     # print(s)
data = np.array(tweets)
print(tweets[0])

prior_days = features_csv['five_prior_close'].tolist()
pdays = []
for days in prior_days:
    string_days = days.strip('][').split(', ')
    for i in range(len(string_days)):
        string_days[i] = float(string_days[i])
    pdays.append(string_days)
# print(pdays[-1])
context = np.array(pdays)
# print(new_context[-1,-1] + new_context[0,0])
'''

for i in range(data.shape[0]):
    data[i] = data[i].lower()
    # print(data[i])

all_text = '$'.join([c for c in data if c not in punctuation])
# print(type(all_text))
reviews_split = all_text.split('$')  # will not work

'''
for j in range(len(reviews_split)):
    reviews_split[j] = reviews_split[j].split(' ')
'''

# '''
all_text2 = ' '.join(reviews_split)
# create a list of words
words = all_text2.split()
# '''

'''
w2v_model = gensim.models.Word2Vec(reviews_split, vector_size=100, window=5, min_count=1, workers=8)

# Retrieve the weights from the model. This is used for initializing the weights
# in a Keras Embedding layer later
w2v_weights = w2v_model.wv.vectors
vocab_size, embedding_size = w2v_weights.shape


# index lookup embedding vector
# try comparing final state and all states

# print("words", reviews_split.shape)
def word2token(word):
    return w2v_model.wv.vocab[word].index


def token2word(token):
    return w2v_model.wv.index2word[token]


for y in range(len(reviews_split)):
    for x in range(len(reviews_split[y])):
        # print(type(reviews_split[y][x]), word2token(reviews_split[y][x]))
        reviews_split[y][x] = word2token(reviews_split[y][x])
        # print(reviews_split[y][x])
'''

# print("words", words.shape)


# '''
# Count all the words using Counter Method
count_words = Counter(words)

total_words = len(words)
print("SPLITHERE", reviews_split)
sorted_words = count_words.most_common(total_words)

vocab_to_int = {w:i for i, (w,c) in enumerate(sorted_words)}
vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
print(vocab_to_int)

reviews_int = []
for review in reviews_split:
    r = [vocab_to_int[w] for w in review.split()]
    reviews_int.append(r)
print(type(reviews_int), reviews_int)
max_words = np.max(np.max(reviews_int))
print(max_words)
max_len = 0
for review in reviews_int:
    if len(review) > max_len:
        max_len = len(review)
print(max_len)

reviews_len = [len(x) for x in reviews_int]
pd.Series(reviews_len).hist()
plt.show()
pd.Series(reviews_len).describe()

# get rid of long and short reviews
# reviews_int = [reviews_int[i] for i, l in enumerate(reviews_len) if l>0 ]
# my_labels = [my_labels[i] for i, l in enumerate(reviews_len) if l> 0 ]
# '''

max_words = 0
max_len = 0
for review in reviews_int:
    for r in review:
        if r > max_words:
            max_words = r
    if len(review) > max_len:
        max_len = len(review)
print(max_len)


def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype=int)

    for i, review in enumerate(reviews_int):
        review_len = len(review)

        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length - review_len))
            new = review + zeroes
        elif review_len > seq_length:
            new = review[0:seq_length]

        features[i, :] = np.array(new)

    return features


features = np.array(pad_features(reviews_int, max_len))  # swap reviews_split out for reviews_int for old
len_feat = len(features)
my_labels = np.array(my_labels)
print("shapes", features.shape, my_labels.shape)

'''
split_frac = 0.6
train_x = features[0:int(split_frac*len_feat)]
train_y = my_labels[0:int(split_frac*len_feat)]
remaining_x = features[int(split_frac*len_feat):]
remaining_y = my_labels[int(split_frac*len_feat):]
valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
valid_y = remaining_y[0:int(len(remaining_y)*0.5)]
test_x = remaining_x[int(len(remaining_x)*0.5):]
test_y = remaining_y[int(len(remaining_y)*0.5):]
# data processed

train_context = context[0:int(split_frac*len_feat)]
remaining_context = context[int(split_frac*len_feat):]
valid_context = remaining_context[0:int(len(remaining_context)*0.5)]
test_context = remaining_context[int(len(remaining_context)*0.5):]
'''

rs = random.randint(0, 100)
print(features.shape, my_labels.shape)
train_x, test_x, train_y, test_y = train_test_split(features, my_labels, test_size=0.2, random_state=rs)
train_context, test_context, train_y, test_y = train_test_split(context, my_labels, test_size=0.2, random_state=rs)

print(train_x.shape, train_context.shape)
in1 = l.Input(shape=[max_len])
in2 = l.Input(shape=[5])

# number of possible words, desired output shape, padded input length
m1 = l.Embedding(max_words, 24, input_length=max_len)(in1)
m1 = l.LSTM(16)(m1)  # can be replaced with SimpleRNN
m1 = l.Dense(8, activation='relu')(m1)

m2 = l.Dense(8, activation='relu')(in2)

m = l.concatenate([m1, m2])
u = l.Dense(1, activation='tanh')(m)

model = k.Model(inputs=[in1, in2], outputs=u)

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit([train_x, train_context], train_y, validation_split=0.25, epochs=10, batch_size=2)
results = model.evaluate([test_x, test_context], test_y)
print("Results: ", results)
pred = model.predict([test_x, test_context])

# model.fit(train_x, train_y, validation_split=0.25, epochs=10, batch_size=2)
# results = model.evaluate(test_x,test_y)
# print("Results: ", results)
# pred = model.predict(test_x)

# model.fit(train_context, train_y, validation_split=0.25, epochs=10, batch_size=2)
# results = model.evaluate(test_context,test_y)
# print("Results: ", results)
# pred = model.predict(test_context)

print('\n')
for item in pred:
    print(item)
print('\n')
for item in test_y:
    print(item)