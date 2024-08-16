import json
import os
import time
from tqdm.auto import tqdm

from arguments import parse_args
from utils import load_target_model, load_prompts, get_recursive_key, PAPTemplate


def inference_on_data(args, data, prompt_key, target_model,
                      defense_method, response_key, target_max_n_tokens=150,
                      existing_response_key=None,
                      max_memory=None, pap_template=False, verbose=False):
    """
    Generate outputs on a list of data.
    Each item in data is a dictionary.
    """

    target_lm = load_target_model(
        args, target_model, target_max_n_tokens, max_memory,
        defense_method=defense_method)
    if pap_template:
        target_lm.template = PAPTemplate(target_lm)

    results = []
    begin = time.time()
    for i, example in enumerate(tqdm(data, disable=verbose)):
        if verbose:
            print('*' * 80)
            print(f'Testing on example {i}')
        if response_key in example and example[response_key] != "":
            print(f"Skipped item {example[prompt_key]} as it already has an output with key {response_key}")
            results.append(example)
            continue
        prompt = get_recursive_key(example, prompt_key)
        assert isinstance(prompt, str)
        if args.reuse_response:
            existing_response = get_recursive_key(example, existing_response_key)
        else:
            existing_response = None

        # inference on target_lm
        response = target_lm.get_response(
            [prompt], responses_list=[existing_response], verbose=verbose)[0]
        assert isinstance(response, str)
        print(f"Reponse: {response}")
        example[response_key] = response
        results.append(example)

        if verbose:
            print('\n' * 2)
    print(f"time cost: {time.time() - begin}")
    return results


def main(args):
    if args.save_result_path is None:
        prompt_path_name = ".".join(args.load_data_path.split("/")[-1].split('.')[:-1])
        args.save_result_path = f"./{prompt_path_name}_{args.target_model}_{args.defense_method}.json"

    if (not args.force_overwrite) and os.path.exists(args.save_result_path):
        raise RuntimeError(f"{args.save_result_path} already exists. "
                           "Set --force_overwrite to overwrite.")

    data = load_prompts(args.load_data_path, args.num_examples,
                        args.starting_index)

    results = inference_on_data(
        args=args,
        data=data,
        prompt_key=args.prompt_key,
        existing_response_key=args.existing_response_key,
        target_model=args.target_model,
        defense_method=args.defense_method,
        response_key=args.response_key,
        target_max_n_tokens=args.target_max_n_tokens,
        max_memory=args.max_memory,
        verbose=args.verbose,
        pap_template=args.pap_template)

    if '/' in args.save_result_path:
        os.makedirs(os.path.dirname(args.save_result_path), exist_ok=True)
    with open(args.save_result_path, 'w+') as f:
        json.dump(results, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    args = parse_args(target_model=True, defense=True)
    main(args)
