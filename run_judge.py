import json
import os
from tqdm.auto import tqdm
from llm_jailbreaking_defense import load_judge
from utils import get_recursive_key
from arguments import parse_args


def judge_on_data(goal_key, inferences, prompt_key, response_key,
                  judge_name, judge_max_n_tokens, judge_temperature,
                  reference_key=None, verbose=False):
    results = []
    for i, inference_item in enumerate(tqdm(
            inferences, desc="Judging on example: ", disable=verbose)):
        if verbose:
            print(f'Judging on example {i}')

        goal = get_recursive_key(inference_item, goal_key)
        prompt = get_recursive_key(inference_item, prompt_key)
        response = get_recursive_key(inference_item, response_key)

        reference = None
        if reference_key is not None:
            print(reference_key)
            reference = get_recursive_key(inference_item, reference_key)

        judge = load_judge(
            judge_name, goal,
            max_n_tokens=judge_max_n_tokens,  temperature=judge_temperature)
        if reference is not None:
            rating = judge.score([prompt], [response], [reference])[0]
        else:
            rating = judge.score([prompt], [response])[0]

        inference_item[f'{judge_name}_rating'] = rating
        results.append(inference_item)

    return results


def main(args):
    if args.save_result_path is None:
        args.save_result_path = args.load_data_path

    if (not args.force_overwrite) and os.path.exists(args.save_result_path):
        raise RuntimeError(f"{args.save_result_path} already exists. "
                           "Set --force_overwrite to overwrite.")

    with open(args.load_data_path, 'r') as f:
        inferences = json.load(f)
        if args.num_examples is not None:
            inferences = inferences[:args.num_examples]

    results = inferences
    for judge_name in args.judge_name:
        results = judge_on_data(goal_key=args.goal_key,
                                inferences=results,
                                prompt_key=args.prompt_key,
                                response_key=args.response_key,
                                judge_name=judge_name,
                                judge_max_n_tokens=args.judge_max_n_tokens,
                                judge_temperature=args.judge_temperature,
                                verbose=args.verbose)
        avg_rating = sum([a[f"{judge_name}_rating"] for a in results])/len(results)
        print(f"Average rating from {judge_name} is {avg_rating}")

    if '/' in args.save_result_path:
        os.makedirs(os.path.dirname(args.save_result_path), exist_ok=True)
    with open(args.save_result_path, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    args = parse_args(judge=True)
    main(args)
