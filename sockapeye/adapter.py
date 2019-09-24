#! /usr/bin/env python3

from sockeye import inference
from sockeye import utils
from contextlib import ExitStack


class SockeyeAdapter:
    def __init__(self, model_path: str) -> None:
        brevity_penalty_weight = 1.0
        brevity_penalty = inference.BrevityPenalty(brevity_penalty_weight)

        with ExitStack() as exit_stack:
            context = utils.determine_context(device_ids=[-1],
                                        use_cpu=True,
                                        disable_device_locking=False,
                                        lock_dir='/tmp',
                                        exit_stack=exit_stack)[0]

        self.models, self.source_vocabs, self.target_vocab = inference.load_models(
            context=context,
            max_input_len=None,
            beam_size=5,
            batch_size=1,
            model_folders=[model_path],
            checkpoints=None,
            softmax_temperature=None,
            max_output_length_num_stds=2,
            decoder_return_logit_inputs=False,
            cache_output_layer_w_b=False,
            override_dtype=None,
            output_scores=False,
            sampling=False)

        self.translator = inference.Translator(context=context,
            ensemble_mode='linear',
            bucket_source_width=10,
            length_penalty=inference.LengthPenalty(1.0,
                                                    0.0),
            beam_prune=0,
            beam_search_stop='all',
            nbest_size=1,
            models=self.models,
            source_vocabs=self.source_vocabs,
            target_vocab=self.target_vocab,
            restrict_lexicon=None,
            avoid_list=None,
            store_beam=False,
            strip_unknown_words=False,
            skip_topk=False,
            sample=None,
            constant_length_ratio=0.0,
            brevity_penalty=brevity_penalty)

    def translate(self, line: str) -> str:
        """

        :param line:
        :return:
        """
        input = inference.make_input_from_plain_string(0, line)
        outputs = self.translator.translate([input])
        return outputs[0].translation
