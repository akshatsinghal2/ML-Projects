from flask import Flask, render_template, request
import pickle

# Load the model and vectorizer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

app = Flask(__name__, static_url_path='/static')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    text = request.form["text"]
    X_input = vectorizer.transform([text])
    language = model.predict(X_input)[0]
    return render_template("result.html", language=language)

if __name__ == "__main__":
    app.run(debug=True)