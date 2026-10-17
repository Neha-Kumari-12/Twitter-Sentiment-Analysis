
import os
import re
import joblib
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "sentiment_model.joblib")
)
vectorizer = joblib.load(
    os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")
)

history = []


def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    # Sentiment140: 0 = Negative, 4 = Positive.
    # Some models encode the positive class as 1 instead of 4.
    if str(prediction) in ("1", "4"):
        return "Positive"
    return "Negative"


@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    tweet = ""

    if request.method == "POST":
        tweet = request.form.get("tweet", "").strip()

        if tweet:
            sentiment = predict_sentiment(tweet)
            history.append({
                "tweet": tweet,
                "sentiment": sentiment
            })

    return render_template(
        "index.html", sentiment=sentiment, tweet=tweet
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/results")
def results():
    positive = sum(
        1 for item in history if item["sentiment"] == "Positive"
    )
    negative = sum(
        1 for item in history if item["sentiment"] == "Negative"
    )

    return render_template(
        "results.html",
        history=history,
        total=len(history),
        positive=positive,
        negative=negative
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
