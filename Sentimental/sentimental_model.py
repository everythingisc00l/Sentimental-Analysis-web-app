from fastai.learner import load_learner
from fastai.vision.all import *
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


# from dostoevsky.tokenization import RegexTokenizer
# from dostoevsky.models import FastTextSocialNetworkModel
# model = FastTextSocialNetworkModel(tokenizer=tokenizer)
# text_input = request.form.get('text_input')
# results = model.predict(text_input, k=2)


@app.route('/show-prediction/', methods=['POST'])
def show_prediction():
    classifier = pipeline(
        model="blanchefort/rubert-base-cased-sentiment-rusentiment")
    text_input = request.form.get('text_input')
    prediction = classifier(text_input)
    class_translation = {
        'NEGATIVE': 'Негативный',
        'NEUTRAL': 'Нейтральный',
        'POSITIVE': 'Позитивный'
    }
    processed_prediction = []
    for item in prediction:
        label = item['label']
        score = item['score']
        translated_label = class_translation.get(label, label)
        processed_prediction.append(
            {'label': translated_label, 'score': score})

    return (render_template('show-prediction.html', prediction=processed_prediction))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
