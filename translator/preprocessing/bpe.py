#!/usr/bin/env python3

"""
Wraps sub-word nmt to provide byte-pair encoding (BPE) as a preprocessing step.
"""
import os

from translator.preprocessing import constants as C
from translator.preprocessing import commander
from translator.preprocessing.external import ExternalProcessor

class BytePairEncoderSegment(object):
    """
    Applies a trained BPE model to individual segments.
    """
    def __init__(self, bpe_model_path, vocab_path=None):
        """
        @param bpe_model_path full path to BPE model
        @param vocab_path optional path to vocabulary file
        """
        arguments = [
            '-c %s' % bpe_model_path
        ]
        if vocab_path is not None:
            arguments.extend([
                '--vocabulary %s' % vocab_path,
                '--vocabulary-threshold %d' % C.BPE_VOCAB_THRESHOLD
            ])

        # the subword script apply_bpe.py needs to be run in a Python 3 environment,
        # a constant is used to avoid version problems
        self._processor = ExternalProcessor(
            command=" ".join([C.PYTHON3] + [C.SUBWORD_NMT_APPLY] + arguments),
            stream_stderr=False,
            trailing_output=False,
            shell=False
        )

    def close(self):
        """
        Deletes reference to obsolete objects.
        """
        del self._processor

    def encode_segment(self, segment):
        """
        Encodes a single @param segment by applying a trained BPE model.
        """
        encoded_segment = self._processor.process(segment)
        return encoded_segment


def bpe_decode_segment(segment):
    """
    Removes byte pair encoding.
    """
    return segment.replace("@@ ", "")