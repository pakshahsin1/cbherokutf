from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import pickle
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Enable CORS for the entire application
CORS(app)

# Initialize the model and other necessary components
interpreter = tflite.Interpreter(model_path='./chatbot_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    response = chatbot_response(user_message)
    return jsonify({'response': response})

@app.route("/")
def home():
    return render_template('index.html')

