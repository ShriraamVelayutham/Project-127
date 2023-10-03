import nltk
nltk.download('punkt')

import json
import pickle
import numpy as np
import random
import tensorflow

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from data_preprocessing import get_stem_words
model=tensorflow.keras.models.load_model("chatbot_model.h5")

ignore_words = ['?', '!',',','.', "'s", "'m"]

train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)
words = pickle.load(open("words.pkl"),"rb")
classes = pickle.load(open("classes.pkl"),"rb")

def prerpocess_user_input(user_input):
    input1 = nltk.word_tokenize(username)
    input2 = get_stem_words(input1, ignore_words)
    input3 = sorted(list(set(input2)))
    bag = []
    bag_of_words = []
    for i in words:
        if i in input2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)

def bot_class_prediction(user_input):
    input1 = prerpocess_user_input(user_input)
    prediction=model.predict(input1)
    prediction_class = np.argmax(prediction[0])
    return(prediction_class)
        
def bot_response(user_input):
    predicted_class_label = bot_class_prediction(user_input)
    predicted_class=classes[predicted_class_label]
    for i in intents['intents']:
        if i['tag'] == predicted_class:
             chatbot_response = random.choice(i["responses"])
             return(chatbot_response)
        
print("Hi, I am chatbot, How may I help you ?")

while True:
    user_input = input("Enter the message here...")
    response = bot_response(user_input)
    print("bot Response",response)

