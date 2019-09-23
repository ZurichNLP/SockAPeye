from flask import Flask
from translator import translate
app = Flask(__name__)
translator = translate.Translator()

@app.route('/')
def index():
    return 'Translation API'

@app.route('/api/translate/<text>')
def api_translate(text):
    return translator.translate_lines([text])[0]

