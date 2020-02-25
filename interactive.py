#! /usr/bin/env python3

import argparse
import logging

from sockapeye import translate
from sockapeye import constants as C


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", "-m", type=str, help="Path to trained Sockeye model",
                        required=True)

    parser.add_argument("--beam-size", type=int, help="Decoding beam size",
                        required=False, default=1)

    parser.add_argument("--steps", type=str, help="Pre-processing steps (postprocessing is determined based on pre-processing)",
                        nargs="+", required=False, default=[], choices=C.PREPROCESS_STEPS)

    parser.add_argument("--src-lang", type=str, help="Source language", required=True)
    parser.add_argument("--trg-lang", type=str, help="Target language", required=True)

    parser.add_argument("--truecase-model", type=str, help="Path to truecasing model",
                        default=None, required=False)
    parser.add_argument("--bpe-model", type=str, help="Path to BPE model",
                        default=None, required=False)

    parser.add_argument("--bpe-vocab-path", type=str, help="Path to BPE vocab for source language",
                        default=None, required=False)
    parser.add_argument("--bpe-vocab-threshold", type=int, help="Vocab threshold to apply for BPE segmentation",
                        default=1, required=False)

    args = parser.parse_args()

    return args


def interactive_loop(translator: translate.Translator) -> None:
    """

    :param translator:
    :return:
    """
    while True:

        line = input(">")

        if line.strip() != "":
            output = translator.translate(line)
            print(output)


def main():

    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    translator = translate.Translator(src_lang=args.src_lang,
                                      trg_lang=args.trg_lang,
                                      model_path=args.model,
                                      truecase_model_path=args.truecase_model,
                                      bpe_model_path=args.bpe_model,
                                      bpe_vocab_path=args.bpe_vocab_path,
                                      bpe_vocab_threshold=args.bpe_vocab_threshold,
                                      processing_steps=args.steps,
                                      beam_size=args.beam_size)

    interactive_loop(translator)



if __name__ == '__main__':
    main()