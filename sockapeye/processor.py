#! /usr/bin/env python3

import sacremoses
import subword_nmt

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
        super(NormalizeStep).__init__(lang, load_from)
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
        super(TokenizeStep).__init__(lang, load_from)
        self.tokenizer = sacremoses.MosesTokenizer(lang=self.lang)

    def process(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        return self.tokenizer.tokenize(line)


class DetokenizeStep(ProcessStep):

    def __init__(self,
                 lang: str,
                 load_from: str = None) -> None:
        """

        """
        super(DetokenizeStep).__init__(lang, load_from)
        self.detokenizer = sacremoses.MosesDetokenizer(lang=self.lang)

    def process(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        return self.detokenizer.detokenize(line)


class TruecaseStep(ProcessStep):

    def __init__(self,
                 lang: str,
                 load_from: str) -> None:
        """

        """
        super(TruecaseStep).__init__(lang, load_from)
        self.truecaser = sacremoses.MosesTruecaser(load_from=self.load_from)

    def process(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        return self.truecaser.truecase(line)


class DetruecaseStep(ProcessStep):

    def __init__(self,
                 lang: str = None,
                 load_from: str = None) -> None:
        """

        """
        super(DetruecaseStep).__init__(lang, load_from)
        self.detruecaser = sacremoses.MosesDetruecaser()

    def process(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        return self.detruecaser.detruecase(line)


class BpeStep(ProcessStep):

    def __init__(self,
                 lang: str,
                 load_from: str = None,
                 vocab: str = None,
                 vocab_threshold: int = None) -> None:
        """

        """
        super(BpeStep).__init__(lang, load_from)

        self.vocab = vocab
        self.vocab_threshold = vocab_threshold

        if vocab is not None:
            vocab = subword_nmt.apply_bpe.read_vocabulary(vocab, vocab_threshold)

        self.bpe_encoder = subword_nmt.apply_bpe.BPE(codes=self.load_from,
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
        super(DebpeStep).__init__(lang, load_from)

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
        for step in self.steps:
            line = step.process(line)

        return line
