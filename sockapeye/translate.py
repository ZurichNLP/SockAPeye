#! /usr/bin/env python3

from . import processor
from . import adapter


class Translator:
    def __init__(self,
                 src_lang: str,
                 trg_lang: str,
                 model_path: str,
                 truecase_model_path: str,
                 bpe_model_path: str,
                 bpe_vocab_path: str = None,
                 bpe_vocab_threshold: int = 50) -> None:
        """

        :param src_lang:
        :param trg_lang:
        :param model_path:
        :param truecase_model_path:
        :param bpe_model_path:
        :param bpe_vocab_path:
        :param bpe_vocab_threshold:
        """
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.model_path = model_path
        self.truecase_model_path = truecase_model_path
        self.bpe_model_path = bpe_model_path
        self.bpe_vocab_path = bpe_vocab_path
        self.bpe_vocab_threshold = bpe_vocab_threshold

        preprocessor_steps = [processor.NormalizeStep(lang=self.src_lang),
                              processor.TokenizeStep(lang=self.src_lang),
                              processor.TruecaseStep(lang=self.src_lang, load_from=self.truecase_model_path),
                              processor.BpeStep(lang=self.src_lang,
                                                load_from=self.bpe_model_path,
                                                vocab=self.bpe_vocab_path,
                                                vocab_threshold=self.bpe_vocab_threshold)]

        self.preprocessor = processor.Processor(steps=preprocessor_steps)

        postprocessor_steps = [processor.DebpeStep(),
                               processor.DetruecaseStep(),
                               processor.DetokenizeStep(lang=self.trg_lang)]

        self.postprocessor = processor.Processor(steps=postprocessor_steps)



        self.model = adapter.SockeyeAdapter(self.model_path)

    def translate(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        line = self.preprocessor.process(line)
        line = self.model.translate(line)
        line = self.postprocessor.process(line)

        return line

