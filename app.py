#! /usr/bin/env python3

from flask import Flask
from sockapeye import translate

model_base = "/Users/mathiasmuller/Desktop/mt19_u6_model"

sample_config = {"src_lang": "de",
                 "trg_lang": "en",
                 "model_path": model_base + "/models/model_wmt17",
                 "truecase_model_path": model_base + "/shared_models/truecase_model.de",
                 "bpe_model_path": model_base + "/shared_models/deen.bpe",
                 "bpe_vocab_path": model_base + "/shared_models/vocab.en",
                 "bpe_vocab_threshold": 50}

app = Flask(__name__)
translator = translate.Translator(**sample_config)

@app.route('/')
def index():
    return 'Translation API'

@app.route('/api/translate/<text>')
def api_translate(text):
    return translator.translate(text)
