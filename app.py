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
    resources={r"/api/predict/*": {"origins": [FRONTEND, FRONTEND_DEV]}},
)

email_cv = pickle.load(open("emailvectorizer.pkl", "rb"))
email_model = pickle.load(open("emailmodel.pkl", "rb"))
message_tfidf = pickle.load(open("messagevectorizer.pkl", "rb"))
message_model = pickle.load(open("messagemodel.pkl", "rb"))

nltk.download("punkt")
nltk.download("stopwords")

ps = PorterStemmer()


def transform_email(text):
    stopwords_set = set(stopwords.words("english"))
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation)).split()
    text = [ps.stem(word) for word in text if word not in stopwords_set]
    text = " ".join(text)
    return text


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


@app.route("/api/predict-email", methods=["POST"])
def predict_email():
    transformed_email = transform_email(request.json["email"])
    vector_input = email_cv.transform([transformed_email])
    result = email_model.predict(vector_input)[0]
    return (
        jsonify({"result": "Spam"}) if result == 1 else jsonify({"result": "Not Spam"})
    )


@app.route("/api/predict-message", methods=["POST"])
def predict_message():
    transformed_sms = transform_text(request.json["message"])
    vector_input = message_tfidf.transform([transformed_sms])
    result = message_model.predict(vector_input)[0]
    return (
        jsonify({"result": "Spam"}) if result == 1 else jsonify({"result": "Not Spam"})
    )


# if __name__ == "__main__":
#     app.run(debug=True)
