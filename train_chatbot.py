import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import warnings
warnings.filterwarnings('ignore')

import random
import json
import pickle

import nltk
nltk.download('all')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

import numpy as np
import pyttsx3

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []

ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

train_x = []
train_y = []

for document in documents:
    # Create the bag-of-words for the input
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    for word in words:
        bag.append(1 if word in word_patterns else 0)
    train_x.append(bag)
    
    # Create the one-hot encoded output row for the label
    output_row = [0] * len(classes)
    output_row[classes.index(document[1])] = 1
    train_y.append(output_row)

# Shuffle the data to ensure the model doesn't learn from the order
combined = list(zip(train_x, train_y))
random.shuffle(combined)
train_x, train_y = zip(*combined)

# Now, convert your lists to NumPy arrays, which will have the correct, uniform shapes
train_x = np.array(train_x)
train_y = np.array(train_y)

model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),),
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01,
          momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)

model.save("chatbot_model.keras")
print("Done!")

# Initialize the text-to-speech engine (do this once, outside the function)
engine = pyttsx3.init()

def predict_class(txt):
    # Dummy implementation for testing
    return "disease_tag"

def get_response(ints, intents_json):
    if len(ints) == 0:
        return "I do not understand. Please try again."
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I do not understand. Please try again."
    
    # Speak the result
    engine.say("Found it. From our Database we found that " + res)
    engine.runAndWait()
    
    # Print the result
    print("Your Symptom was  : ", txt)
    print("Result found in our Database : ", res) 