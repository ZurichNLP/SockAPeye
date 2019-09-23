from translator.preprocessing.normalizer import Normalizer
from translator.preprocessing.tokenizer import Tokenizer, Detokenizer
from translator.preprocessing.truecaser import Truecaser, Detruecaser
from translator.preprocessing.bpe import BytePairEncoderSegment, bpe_decode_segment
from translator.preprocessing.external import ExternalProcessor
from translator.sockeyeadapter import SockeyeAdapter

class Translator:
    def __init__(self):
        # Define some parameters - for now, hard-coded
        self.src_lang = "de"
        self.trg_lang = "en"
        self.model_base = "/Users/raphael/projects/sockeye-toy-models/mt19_u6_model"
        self.normalizer = Normalizer(self.src_lang)
        self.tokenizer = Tokenizer(self.src_lang)
        self.detokenizer = Detokenizer(self.trg_lang)

        truecase_model = self.model_base + "/shared_models/truecase_model." + self.src_lang
        self.truecaser = Truecaser(truecase_model)
        self.detruecaser = Detruecaser()

        bpe_model = self.model_base + "/shared_models/" + self.src_lang + self.trg_lang + ".bpe"
        bpe_vocab = self.model_base + "/shared_models/vocab." + self.src_lang
        self.bpe = BytePairEncoderSegment(bpe_model, bpe_vocab)

        self.sockeye = SockeyeAdapter(self.model_base + "/models/model_wmt17")

    def translate_lines(self, input_lines):
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