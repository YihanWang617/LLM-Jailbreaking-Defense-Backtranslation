import json
import os
from llm_jailbreaking_defense import load_defense, TargetLM, DefendedTargetLM
from llm_jailbreaking_defense.defenses import args_to_defense_config


def load_target_model(args, target_model, target_max_n_tokens=150,
                      max_memory=None, preloaded_model=None, defense_method=None,
                      add_system_prompt=True):

    target_preloaded_model = defense_preloaded_model = None

    if hasattr(args, "attack_model") and args.attack_model == args.target_model:
        print("Using same attack and target model. Using previously loaded model.")
        target_preloaded_model = preloaded_model
    targetLM = TargetLM(model_name=target_model,
                        max_n_tokens=target_max_n_tokens,
                        max_memory=max_memory,
                        preloaded_model=target_preloaded_model,
                        add_system_prompt=add_system_prompt)

    if defense_method is not None:
        if hasattr(args, "attack_model") and args.attack_model == args.backtranslation_infer_model:
            defense_preloaded_model = preloaded_model
        if args.target_model == args.backtranslation_infer_model:
            defense_preloaded_model = targetLM.model

        defense_config = args_to_defense_config(args)
        defense = load_defense(defense_config, defense_preloaded_model)
        targetLM = DefendedTargetLM(target_model=targetLM,
                                    defense=defense)

    return targetLM


def load_prompts(input_path, num_examples=None, offset=None):
    prompts = []

    def _read_file(filename):
        with open(filename, 'r') as f:
            prompts.extend(json.load(f))

    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            _read_file(os.path.join(input_path, filename))
    else:
        _read_file(input_path)

    if offset:
        prompts = prompts[offset:]
    if num_examples:
        prompts = prompts[:num_examples]

    print(f'{len(prompts)} examples loaded.')

    return prompts


def get_recursive_key(obj, key, delim='.'):
    # resursively get attr for nested object
    for k in key.split(delim):
        if k not in obj:
            return None
        obj = obj[k]
    return obj
