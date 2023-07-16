import string
import numpy as np
import re
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import pydot

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_train):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents

# load and clean a dataset
def load_clean_dataset(vocab, is_train):
    # load documents
    neg = process_docs(r'C:\Users\mslus\Desktop\Projects\rafinacja-zaliczenie\neg', vocab, is_train)
    pos = process_docs(r'C:\Users\mslus\Desktop\Projects\rafinacja-zaliczenie\pos', vocab, is_train)
    docs = neg + pos
    # prepare labels
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
    return docs, labels

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def pad_sequences(sequences, maxlen=None, padding_value=0):
    # Find the maximum sequence length if not provided
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        # Pad the sequence with the padding value
        padded_seq = seq[:maxlen] + [padding_value] * max(0, maxlen - len(seq))
        padded_sequences.append(padded_seq)
    return padded_sequences

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences
    padded = pad_sequences(encoded, maxlen=max_length)
    return padded

# define the model
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load training data
train_docs, y_treningowe = load_clean_dataset(vocab, True)
# create the tokenizer
tokenizer = create_tokenizer(train_docs)
# define vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
# calculate the maximum sequence length
max_length = max([len(s.split()) for s in train_docs])
print('Maximum length: %d' % max_length)
# encode data
x_treningowe = encode_docs(tokenizer, max_length, train_docs)
# define model
model = define_model(vocab_size, max_length)
# fit network

# to array
x_treningowe = np.array(x_treningowe)
y_treningowe = np.array(y_treningowe)

model.fit(x_treningowe, y_treningowe, epochs=5, verbose=2)
# save the model

##    Save Model to JSON
##    Save Model to YAML
## -> Save Model to HDF5

model.save('model.h5')

# evaluate model on test dataset
# _, acc = model.evaluate(Xtest, ytest, verbose=0)
# print('Test Accuracy: %.2f' % (acc*100))

# # test positive text
# text = 'Everyone will enjoy this film. I love it, recommended!'
# percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
# print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))

# # test negative text
# text = 'This is a bad movie. Do not watch it. It sucks.'
# percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
# print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
