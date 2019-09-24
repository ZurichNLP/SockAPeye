#! /usr/bin/env python3

import copy
import logging

from sockapeye import translate

class TranslatorCache:

    def __init__(self, base_config):
        self.base_config = base_config
        self.cache = {}

    def get_or_create(self, src_lang, trg_lang):
        cache_key = (src_lang, trg_lang)
        if cache_key not in self.cache:
            config = copy.deepcopy(self.base_config)
            config["src_lang"] = src_lang
            config["trg_lang"] = trg_lang
            logging.debug(str(config))
            self.cache[cache_key] = translate.Translator(**config)

        return self.cache[cache_key]
        