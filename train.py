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
# add the embedding layer used to set up and update weights for each of the
# integer vectors(words). Weights can be retrieved by multiplying the one-hot
# vector assigned to each integer vector to the embedding matrix.
model.add(Embedding(1000, 16, input_length=20))
# ( Need to revisit this )
model.add(GlobalAveragePooling1D())
# a neural network layer is added to the model that creates a weights matrix.
# This layer implements the output = activation(dot(input, kernel) + bias
# operation where the kernel is the weights matrix and the activation is the
# rectified linear activation function.
model.add(Dense(16, activation='relu'))
# another neural network layer is added same as before
model.add(Dense(16, activation='relu'))
# another neural network is added, but the activation function is softmax
# which converts the previous output to a vector of categorical probabilities
model.add(Dense(num_labels, activation='softmax'))
# ( need to revisit this) sets up the configurations for the neural network
# model. The Adam gradient descent method is used.
# sparse_categorical_crossentropy is used to get the crossentropy loss between
# predictions and labels. accuracy in metrics is used to get how often labels
# are predicted corrrectly
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# a summary of the neural network gets printed
model.summary()

# the neural network model gets trained for 500 iterations
# and a record of loss and metric values is given with a History object
# returned
history = model.fit(padded_sequences, np.array(training_labels), epochs=500)

# the neural network model gets saved to TensorFlow SavedModel
model.save("chat_model")

# the tokenizer used to vectorize the data gets saved using the pickle module
with open('tokenizer.pickle', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)

# the label encoder with the encoded target labels gets saved as well
with open('label_encoder', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file, protocol=pickle.HIGHEST_PROTOCOL)

