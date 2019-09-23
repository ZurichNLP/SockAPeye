#! /usr/bin/env python3

from . import processor
from . import adapter

class Translator:
    def __init__(self):
        # Define some parameters - for now, hard-coded
        self.src_lang = "de"
        self.trg_lang = "en"
        self.model_base = "/Users/raphael/projects/sockeye-toy-models/mt19_u6_model"
        truecase_model = self.model_base + "/shared_models/truecase_model." + self.src_lang

        bpe_model = self.model_base + "/shared_models/" + self.src_lang + self.trg_lang + ".bpe"
        bpe_vocab = self.model_base + "/shared_models/vocab." + self.src_lang
        vocab_threshold = 50

        preprocessor_steps = [processor.NormalizeStep(lang=self.src_lang),
                              processor.TokenizeStep(lang=self.src_lang),
                              processor.TruecaseStep(lang=self.src_lang, load_from=truecase_model),
                              processor.BpeStep(lang=self.src_lang,
                                                load_from=bpe_model,
                                                vocab=bpe_vocab,
                                                vocab_threshold=vocab_threshold)]

        self.preprocessor = processor.Processor(steps=preprocessor_steps)



        self.sockeye = SockeyeAdapter(self.model_base + "/models/model_wmt17")

    def translate_lines(self, lines):

        for line in lines:
            line = self.preprocessor.process(line)
            line = self.translator.translate(line)
            line = self.postprocessor.process(line)

            yield line

        lines = map(self.preprocess, input_lines)
        
        lines = map(self.translate, lines)
        print(list(lines))
        lines = map(self.postprocess, lines)
        return list(lines)

    def preprocess(self, line):
        # Normalize
        line = self.normalizer.normalize_punctuation(line)
        # Tokenize
        line = self.tokenizer.tokenize(line, split=False)
        # Truecasing
        line = self.truecaser.truecase_segment(line)
        # BPE
        line = self.bpe.encode_segment(line)
        return line

    def postprocess(self, line):
        # De-BPE
        line = bpe_decode_segment(line)
        # De-Truecase
        line = self.detruecaser.detruecase_segment(line)
        # De-Tokenize
        line = self.detokenizer.detokenize(line)
        return line

    def translate(self, input):
        return self.sockeye.translate(input)
    

# # Temporary testing
# translator = Translator()
# translator.preprocess_line("Hallo")