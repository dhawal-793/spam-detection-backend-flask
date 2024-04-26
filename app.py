from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from os import environ

FRONTEND = environ["FRONTEND"]
FRONTEND_DEV = environ["FRONTEND_DEV"]

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/predict": {
            "origins": [FRONTEND,FRONTEND_DEV]
        }
    },
)

tfidf = pickle.load(open("myvectorizer.pkl", "rb"))
model = pickle.load(open("mymodel.pkl", "rb"))

nltk.download("punkt")
nltk.download("stopwords")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    list = []
    for i in text:
        if i.isalnum():
            list.append(i)
    text = list[:]
    list.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            list.append(i)
    text = list[:]
    list.clear()
    for i in text:
        list.append(ps.stem(i))
    return " ".join(list)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Hello form server"})


@app.route("/api/predict", methods=["POST"])
def predict():
    transformed_sms = transform_text(request.json["message"])
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    return jsonify({"result": "Spam"}) if result==1 else jsonify({"result": "NotSpam"})

