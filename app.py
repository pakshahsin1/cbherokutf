import os
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import json
import pickle
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
import tflite_runtime.interpreter as tflite

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path='chatbot_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load data
intents = json.loads(open('buttons.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, interpreter):
    p = bow(sentence, words, show_details=False).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], [p])
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]['index'])[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, interpreter)
    res = getResponse(ints, intents)
    return res

start = True
while start:
    query = input('Enter Message:')
    if query in ['quit', 'exit', 'bye']:
        start = False
        continue
    try:
        res = chatbot_response(query)
        print(res)
    except Exception as e:
        print('You may need to rephrase your question.')
        print(f"Error: {e}")
