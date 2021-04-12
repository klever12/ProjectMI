# resource:
# https://towardsdatascience.com/how-to-build-your-own-chatbot-using-deep-learning-bb41f970e281

import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# get the data from the JSON file
with open('intents.json') as file:
    data = json.load(file)

# the sample messages(patterns) for each intent
training_patterns = []
# the target labels (tags)
training_labels = []
# the labels
labels = []
# the responses
responses = []

# iterate through each of the indents in the data
for intent in data['intents']:
    # iterate through each pattern in the patterns of the intent
    for pattern in intent['patterns']:
        # add pattern to array
        training_patterns.append(pattern)
        # add corresponding label(tag) of pattern to array
        training_labels.append(intent['tag'])
    # add the responses to the responses array
    responses.append(intent['responses'])

    # check if the tag is already in the labels array
    if intent['tag'] not in labels:
        # add the label(tag) to the labels array
        labels.append(intent['tag'])

# the number of labels
num_labels = len(labels)

# set up the label encoder that will encodes target labels with values between
# 0 - # of labels - 1
label_encoder = LabelEncoder()
# fit the label encoder and get the encoded labels
training_labels = label_encoder.fit_transform(training_labels)

# set up the "Tokenizer" class that will be used to vectorize the text data
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV")
# update the internal vocabulary based on the list of messages(patterns)
# index dictionary is created for every word, and they get a unique integer
# based on frequency.
tokenizer.fit_on_texts(training_patterns)
# get the dictionary of words and their unique integer
word_index = tokenizer.word_index
# each word in the text gets replaced with the corresponding unique integer
# value
sequences = tokenizer.texts_to_sequences(training_patterns)
# takes the sequences and transforms it into a 2D Numpy array where each row
# is the message(pattern). Max length for each sequence is 20, and messages
# bigger gets values removed at the end.
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=20)

# set up and define the neural network model
model = Sequential()



