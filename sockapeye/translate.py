#! /usr/bin/env python3

import time
import logging

from typing import List

from . import processor
from . import adapter
from . import constants as C


class Translator:
    def __init__(self,
                 src_lang: str,
                 trg_lang: str,
                 model_path: str,
                 truecase_model_path: str,
                 bpe_model_path: str,
                 bpe_vocab_path: str = None,
                 bpe_vocab_threshold: int = 50,
                 processing_steps: List[str] = None,
                 beam_size:int = 1) -> None:
        """

        :param src_lang:
        :param trg_lang:
        :param model_path:
        :param truecase_model_path:
        :param bpe_model_path:
        :param bpe_vocab_path:
        :param bpe_vocab_threshold:
        :param processing_steps:
        """

        tic = time.time()

        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.model_path = model_path
        self.truecase_model_path = truecase_model_path
        self.bpe_model_path = bpe_model_path
        self.bpe_vocab_path = bpe_vocab_path
        self.bpe_vocab_threshold = bpe_vocab_threshold
        self.beam_size = beam_size

        preprocessor_steps = []
        postprocessor_steps = []

        if C.PREPROCESS_STEP_NORM in processing_steps:
            preprocessor_steps.append(processor.NormalizeStep(lang=self.src_lang))

        if C.PREPROCESS_STEP_TOK in processing_steps:
            preprocessor_steps.append(processor.TokenizeStep(lang=self.src_lang))
            postprocessor_steps.insert(0, processor.DetokenizeStep(lang=self.trg_lang))

        if C.PREPROCESS_STEP_TRU in processing_steps:
            preprocessor_steps.append(processor.TruecaseStep(lang=self.src_lang, load_from=self.truecase_model_path))
            postprocessor_steps.insert(0, processor.DetruecaseStep())

        if C.PREPROCESS_STEP_BPE in processing_steps:
            preprocessor_steps.append(
                processor.BpeStep(lang=self.src_lang,
                                  load_from=self.bpe_model_path,
                                  vocab=self.bpe_vocab_path,
                                  vocab_threshold=self.bpe_vocab_threshold))
            postprocessor_steps.insert(0, processor.DebpeStep())

        self.preprocessor = processor.Processor(steps=preprocessor_steps)
        self.postprocessor = processor.Processor(steps=postprocessor_steps)

        self.model = adapter.SockeyeAdapter(self.model_path, self.beam_size)

        toc = tic - time.time()

        logging.debug("Loading models and preparing translator took: %f s" % toc)

    def translate(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        line = self.preprocessor.process(line)

        tic = time.time()
        line = self.model.translate(line)
        toc = time.time()

        logging.debug("Translation time: %f seconds" % (toc - tic))

        line = self.postprocessor.process(line)

        return line

