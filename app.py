from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from os import environ

FRONTEND_DEV = environ.get("FRONTEND_DEV")
FRONTEND_BUILD = environ.get("FRONTEND_BUILD")
FRONTEND = environ["FRONTEND"]

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/predict": {
            "origins": [
                FRONTEND,
                FRONTEND_DEV,
                FRONTEND_BUILD,
            ]
        }
    },
)

# Load the pre-trained model and vectorizer
tfidf = pickle.load(open("myvectorizer.pkl", "rb"))
model = pickle.load(open("mymodel.pkl", "rb"))

# Initialize NLTK
nltk.download("punkt")
nltk.download("stopwords")

# Initialize Porter Stemmer
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route("/", methods=["GET"])
def home():
    response = jsonify({"message": "Hello form server"})
    return response


@app.route("/api/predict", methods=["POST"])
def predict():
    # Get input message from request
    input_sms = request.json["message"]
    # print("Given Input", input_sms)
    # Preprocess input message
    transformed_sms = transform_text(input_sms)
    # print("Transformed Input", transformed_sms)
    # Vectorize input message
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]
    # print(result)

    # Return prediction result as JSON response
    response = ""
    if result == 1:
        response = jsonify({"result": "Spam"})
    else:
        response = jsonify({"result": "Not Spam"})
    return response


if __name__ == "__main__":
    app.run(debug=True)
