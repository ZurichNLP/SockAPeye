#! /usr/bin/env python3

import sacremoses
import logging

from subword_nmt import apply_bpe
from typing import List


class ProcessStep(object):

    def __init__(self,
                 lang: str,
                 load_from: str) -> None:
        """

        """
        self.lang = lang
        self.load_from = load_from

    def process(self, line: str) -> str:
        raise NotImplementedError


class NormalizeStep(ProcessStep):

    def __init__(self,
                 lang: str,
                 load_from: str = None) -> None:
        """

        """
        super(NormalizeStep, self).__init__(lang, load_from)
        self.normalizer = sacremoses.MosesPunctNormalizer(lang=self.lang)

    def process(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        return self.normalizer.normalize(line)


class TokenizeStep(ProcessStep):

    def __init__(self,
                 lang: str,
                 load_from: str = None) -> None:
        """

        """
        super(TokenizeStep, self).__init__(lang, load_from)
        self.tokenizer = sacremoses.MosesTokenizer(lang=self.lang)

    def process(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        return self.tokenizer.tokenize(line, return_str=True)


class DetokenizeStep(ProcessStep):

    def __init__(self,
                 lang: str,
                 load_from: str = None) -> None:
        """

        """
        super(DetokenizeStep, self).__init__(lang, load_from)
        self.detokenizer = sacremoses.MosesDetokenizer(lang=self.lang)

    def process(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        tokens = line.split(" ")
        return self.detokenizer.detokenize(tokens)


class TruecaseStep(ProcessStep):

    def __init__(self,
                 lang: str,
                 load_from: str) -> None:
        """

        """
        super(TruecaseStep, self).__init__(lang, load_from)
        self.truecaser = sacremoses.MosesTruecaser(load_from=self.load_from)

    def process(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        return self.truecaser.truecase(line, return_str=True)


class DetruecaseStep(ProcessStep):

    def __init__(self,
                 lang: str = None,
                 load_from: str = None) -> None:
        """

        """
        super(DetruecaseStep, self).__init__(lang, load_from)
        self.detruecaser = sacremoses.MosesDetruecaser()

    def process(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        return self.detruecaser.detruecase(line, return_str=True)


class BpeStep(ProcessStep):

    def __init__(self,
                 lang: str,
                 load_from: str = None,
                 vocab: str = None,
                 vocab_threshold: int = None) -> None:
        """

        """
        super(BpeStep, self).__init__(lang, load_from)

        self.vocab = vocab
        self.vocab_threshold = vocab_threshold

        load_from_handle = open(self.load_from, "r")

        if vocab is not None:
            vocab_handle = open(vocab, "r")
            vocab = apply_bpe.read_vocabulary(vocab_handle, vocab_threshold)

        self.bpe_encoder = apply_bpe.BPE(codes=load_from_handle,
                                         merges=-1,
                                         separator='@@',
                                         vocab=vocab,
                                         glossaries=None)

    def process(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        return self.bpe_encoder.process_line(line)


class DebpeStep(ProcessStep):

    def __init__(self,
                 lang: str = None,
                 load_from: str = None) -> None:
        """

        """
        super(DebpeStep, self).__init__(lang, load_from)

    def process(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        return line.replace("@@ ", "")


class Processor(object):

    def __init__(self,
                 steps: List[ProcessStep]) -> None:
        """

        :param steps:
        """
        self.steps = steps

    def process(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        logging.debug("Before processing: '%s'" % line)

        for step in self.steps:
            line = step.process(line)
            logging.debug("After step '%s': '%s'" % (step.__class__.__name__, line))

        return line
