import argparse
import math


def add_judge_args(parser):
    parser.add_argument('--judge_name', nargs='+',
                        default=['pair@gpt-4'],
                        help='name of the judge to be used: [judge_method]@[judge_model]')
    parser.add_argument('--judge_max_n_tokens', type=int, default=10,
                        help='Maximum number of tokens for the judge.')
    parser.add_argument('--judge_temperature', type=float, default=0,
                        help='Temperature to use for judge.')


def add_key_args(parser):
    parser.add_argument('--goal_key', type=str, default='goal')
    parser.add_argument('--target_key', type=str, default='target')
    parser.add_argument('--prompt_key', type=str, default='jailbreaking')
    parser.add_argument('--response_key', type=str, default='output')
    parser.add_argument('--reference_key', type=str, default=None)
    parser.add_argument('--jailbreaking_key', type=str, default='jailbreaking')
    parser.add_argument('--existing_response_key', type=str, default='response')
    parser.add_argument('--reuse_response', action='store_true')


def add_target_model_args(parser):
    parser.add_argument('--target_model', type=str, default='vicuna',
                        help='Name of target model.',
                        choices=['vicuna', # by default, it's "lmsys/vicuna-13b-v1.5"
                                 'vicuna-13b-v1.5',
                                 'gpt-4',
                                 'gpt-3.5-turbo', # by default, it's "gpt-3.5-turbo-0613"
                                 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-0613',
                                 'llama-2',  # by default, it's "meta-llama/Llama-2-7b-chat-hf"
                                 'llama-2-13b',
                        ])
    parser.add_argument('--target_max_n_tokens', type=int, default=150,
                        help='Maximum number of generated tokens for the target.')
    parser.add_argument('--max_memory', type=int, default=None)
    parser.add_argument('--target_model_batch_size', type=int, default=2)
    parser.add_argument('--pap_template', action='store_true')


def add_defense_args(parser):
    parser.add_argument('--defense_method', type=str, default='None',
                        help='Defense method applied on the target model. '
                        'No defense by default, choose from "None", "SmoothLLM", '
                        '"backtranslation", '
                        '"backtranslation_with_threshold_[a negative float numer]", '
                        '"response_check", '
                        '"paraphrase_prompt"]')
    parser.add_argument('--backtranslation_threshold', type=float, default=-2.0)
    parser.add_argument('--backtranslation_new_response_length', type=int, default=None)
    parser.add_argument('--backtranslation_infer_model', type=str, default='vicuna')
    parser.add_argument('--paraphrase_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--return_new_response_anyway', action='store_true') # turned off by default
    parser.add_argument('--response_check_threshold', type=int, default=5)
    parser.add_argument('--SmoothLLM_perturbation_num_samples', type=int, default=3)
    parser.add_argument('--SmoothLLM_perturbation_ratio', type=float, default=0.1)
    parser.add_argument('--SmoothLLM_perturbation_type', type=str, default='swap')


def add_data_args(parser):
    parser.add_argument('--load_data_path', type=str)
    parser.add_argument('--save_result_path', type=str)
    parser.add_argument('--num_examples', type=int, default=None,
                        help='Number of examples to use. By default, use all the provided examples.')
    parser.add_argument('--starting_index', '--start', type=int, default=0,
                        help='Index of the starting example.')


def parse_args(target_model=False, defense=False, judge=False):
    parser = get_parser(target_model, defense, judge)

    args = parser.parse_args()
    print('Arguments:', args)
    return args

def get_parser(target_model=False, defense=False, judge=False):
    parser = argparse.ArgumentParser()

    add_data_args(parser)
    add_key_args(parser)
    if target_model:
        add_target_model_args(parser)
    if defense:
        add_defense_args(parser)
    if judge:
        add_judge_args(parser)

    # Misc
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--force_overwrite', action='store_true')

    return parser
