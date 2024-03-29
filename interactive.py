#! /usr/bin/env python3

import argparse
import logging
import readline

from sockapeye import translate
from sockapeye import constants as C


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", "-m", type=str, help="Path to trained Sockeye model",
                        required=True)

    parser.add_argument("--beam-size", type=int, help="Decoding beam size",
                        required=False, default=1)

    parser.add_argument("--quiet", "-q", action="store_true", help="no debug output or logging",
                        required=False, default=False)

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


def parse_execute_escape_cmd(translator: translate.Translator,
                             line: str) -> bool:
    """

    :param translator:
    :param line:
    :return:
    """
    parts = line.split(" ")

    try:
        _, cmd = parts
    except ValueError:
        print("Did not understand your escape command, sorry!")
        return False

    if cmd not in C.ESCAPE_CMDS:
        print("Did not understand your escape command '%s', sorry!" % cmd)
        return False

    if cmd == C.ESCAPE_CMD_QUIET:

        logging_level = logging.CRITICAL
        logging.getLogger().setLevel(logging_level)

        return True

    if cmd == C.ESCAPE_CMD_VERBOSE:
        logging_level = logging.DEBUG
        logging.getLogger().setLevel(logging_level)

        return True

    if cmd == C.ESCAPE_CMD_REMOVE:
        translator.disable_processing()
        return True

    if cmd == C.ESCAPE_CMD_ADD:
        translator.enable_processing()
        return True


def interactive_loop(translator: translate.Translator) -> None:
    """

    :param translator:
    :return:
    """
    while True:

        try:
            line = input("> ")

            if line.startswith(C.ESCAPE_CHAR):
                parse_execute_escape_cmd(translator, line)
                continue

            if line.strip() != "":
                output = translator.translate(line)
                print("  " + output)

        except KeyboardInterrupt:
            print()
            exit(0)


def main():

    args = parse_args()

    if args.quiet:
        logging_level = logging.CRITICAL
    else:
        logging_level = logging.DEBUG

    logging.basicConfig(level=logging_level)

    if C.PREPROCESS_STEP_TRU in args.steps:
        assert args.truecase_model is not None

    if C.PREPROCESS_STEP_BPE in args.steps:
        assert args.bpe_model is not None

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
