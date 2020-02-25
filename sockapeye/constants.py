#! /usr/bin/env python3

PREPROCESS_STEP_TOK = "tokenize"
PREPROCESS_STEP_NORM = "normalize"
PREPROCESS_STEP_TRU = "truecase"
PREPROCESS_STEP_BPE = "bpe"

PREPROCESS_STEPS = [PREPROCESS_STEP_TOK,
                    PREPROCESS_STEP_NORM,
                    PREPROCESS_STEP_TRU,
                    PREPROCESS_STEP_BPE]