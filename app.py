#! /usr/bin/env python3

import logging

from flask import Flask, request
from flask_cors import CORS
from sockapeye import translate
import json
from os import getenv, environ
from sockapeye.languages import LanguageResolver

logging.basicConfig(level=logging.DEBUG)

model_base = getenv("MODEL_BASE")
if not model_base:
    model_base = "/Users/raphael/projects/sockeye-toy-models/mt19_u6_model"

sample_config = {"src_lang": "de",
                 "trg_lang": "en",
                 "model_path": model_base + "/models/model_wmt17",
                 "truecase_model_path": model_base + "/shared_models/truecase_model.de",
                 "bpe_model_path": model_base + "/shared_models/deen.bpe",
                 "bpe_vocab_path": model_base + "/shared_models/vocab.en",
                 "bpe_vocab_threshold": 50}

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

translator = translate.Translator(**sample_config)
languages = LanguageResolver()

@app.route('/')
def index():
    return 'Translation API'

# Route for testing api directly (without front-end, get request)
@app.route('/api/translate/<text>', methods = ['GET'])
def api_translate_get(text):
    return translator.translate(text)

# Route for front-end (post)
@app.route('/api/translate', methods = ['POST'])
def api_translate():
    translation = translator.translate(request.get_json().get('text'))
    return json.dumps({ 'translation': translation })

# Allow front-end to get all available language pairs
@app.route('/api/languages', methods = ['GET'])
def api_languages():
    return json.dumps(languages.available_languages())

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(environ.get('PORT', 5000))
    with app.app_context():
        app.run(threaded=False,
                debug=True,
                port=port)
