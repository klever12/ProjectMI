import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
colorama.init()
from colorama import Fore, Style, Back
import random
import pickle

# read the data from teh json file
with open("intents.json") as file:
    data = json.load(file)

# this function starts the chat between the bot and the user, beginning with
# the option for the user to input a question. The user may stop the program
# by typing quit
def chat():
    # load the model trained in train.py
    model = keras.models.load_model('chat_model')

    # a summary of the neural network gets printed
    model.summary()

    # load the tokenizer object saved in train.py

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load the label encoder object saved in train.py
    with open('label_encoder', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameter for pad_sequences. Makes all sequences have length 20
    max_len = 20

    # while the user is still communicating with the bot (they have not typed
    # quit), the user will be able to input a question and the bot will respond
    while True:
        # "User:" will appear on the screen in blue. The user can type there,
        # and their input will be saved unless 'quit' is typed in which case the
        # program will end
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL)
        inp = input()
        if inp.lower() == "quit":
            break

        # Uses the model from train.py to predict the intent of the user
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        # once the correct intent is found, a response to the question is chosen
        # and printed as the bot response
        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))

# the program starts with the message "Start messaging with the bot (type quit
# to stop)!", then begins the chat functionality
print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
