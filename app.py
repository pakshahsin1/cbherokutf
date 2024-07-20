import os
import requests
import json
import pickle
import numpy as np
import random
from flask import Flask, request, jsonify
from nltk.stem import WordNetLemmatizer
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "Chatbot API is running"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    response = chatbot_response(user_message)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

# Your existing chatbot code...

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

model_url = 'https://drive.google.com/file/d/1-5hFH-yyrgvKedGiIrheze4Mm-htUlUr' # h5
words_url = 'https://drive.google.com/file/d/1JxNE-vLJEHXuTGykz4GfiS_BCoxJdR8_/view?usp=sharing'  #word.pkl
classes_url = 'drive.google.com/file/d/1zy-1Dhu75hsqUBZ0ahCq1WwHEj6Ximdf'  # Classes.pkl
buttons_url = 'https://drive.google.com/file/d/1xcJl9LwuPtC8mOXecUyEOJlxv2hK6iC3/view?usp=sharing'  # buttons.json

if not os.path.exists('chatbot_model.h5'):
    download_file(model_url, 'chatbot_model.h5')
if not os.path.exists('words.pkl'):
    download_file(words_url, 'words.pkl')
if not os.path.exists('classes.pkl'):
    download_file(classes_url, 'classes.pkl')
if not os.path.exists('buttons.json'):
    download_file(buttons_url, 'buttons.json')

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

def download_nltk_packages(file_path):
    with open(file_path, 'r') as file:
        packages = file.read().splitlines()
        for package in packages:
            package = package.strip()
            nltk.download(package)

nltk_data_file = 'nltk.txt'
download_nltk_packages(nltk_data_file)

interpreter = tflite.Interpreter(model_path='chatbot_model.tflite')
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
