import sockeye.inference as inference

from sockeye.utils import determine_context

class SockeyeAdapter:
    def __init__(self, model_path):
        brevity_penalty_weight = 1.0
        brevity_penalty = inference.BrevityPenalty(brevity_penalty_weight)
        context = determine_context(device_ids=[-1],
                                    use_cpu=True,
                                    disable_device_locking=False,
                                    lock_dir='/tmp',
                                    exit_stack=None)[0]

        self.models, self.source_vocabs, self.target_vocab = inference.load_models(
            context=context,
            max_input_len=None,
            beam_size=10,
            batch_size=100,
            model_folders=[model_path],
            checkpoints=None,
            softmax_temperature=None,
            max_output_length_num_stds=2,
            decoder_return_logit_inputs=False,
            cache_output_layer_w_b=False,
            override_dtype=None,
            output_scores=False,
            sampling=None)

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

    # Arguments: Namespace(avoid_list=None, batch_size=100, beam_prune=0, beam_search_stop='all', beam_size=10, brevity_penalty_constant_length_ratio=0.0, brevity_penalty_type='none', brevity_penalty_weight=1.0, bucket_width=10, checkpoints=None, chunk_size=None, config=None, device_ids=[-1], disable_device_locking=False, ensemble_mode='linear', input=None, input_factors=None, json_input=False, length_penalty_alpha=1.0, length_penalty_beta=0.0, lock_dir='/tmp', loglevel='INFO', max_input_len=None, max_output_length_num_stds=2, models=['/Users/raphael/projects/sockeye-toy-models/mt19_u6_model/models/model_wmt17'], nbest_size=1, output=None, output_type='translation', override_dtype=None, quiet=False, restrict_lexicon=None, restrict_lexicon_topk=None, sample=None, seed=None, skip_topk=False, softmax_temperature=None, strip_unknown_words=False, sure_align_threshold=0.9, use_cpu=True)

    def translate(self, input):
        inputs = inference.make_input_from_plain_string(1, input)
        outputs = self.translator.translate([inputs])
        return outputs[0].translation