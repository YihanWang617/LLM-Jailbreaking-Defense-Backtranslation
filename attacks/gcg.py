"""Script for processing GCG's input and output."""

import argparse
import json
import os


def process(input_file, transfer=None):
    print('Input:', input_file)

    with open(input_file, 'r') as file:
        res_ori = json.loads(file.read())
    if transfer:
        print('Transfer attack:', transfer)
        suffix_transfer = ''
        for t in transfer:
            if suffix_transfer != '':
                suffix_transfer += ' '
            with open(t, 'r') as file:
                suffix_ = json.loads(
                    file.read())['controls'][-1]
                print(f'Suffix from {t}: {suffix_}\n')
                suffix_transfer += suffix_
        print('Concatenated suffix:', suffix_transfer)
    else:
        suffix_transfer = None
    config = res_ori['params']
    total_steps = config['n_steps']
    test_steps = config.get('test_steps', 50)
    log_steps = total_steps // test_steps + 1
    tests = []
    controls = []
    for i in range(len(config['goals'])):
        idx = (i + 1) * log_steps - 1
        tests.append(res_ori['tests'][idx])
        controls.append(res_ori['controls'][idx])
    results = []
    success = 0
    for i, test in enumerate(tests):
        if test['n_passed'][0]:
            success += 1
        example = {
            'goal': config['goals'][i],
            'target': config['targets'][i],
        }
        if suffix_transfer:
            example['jailbreaking'] = config['goals'][i] + ' ' + suffix_transfer
        else:
            example['jailbreaking'] = config['goals'][i] + ' ' + controls[i]
        results.append(example)
    print(f"Attacked according to GCG's code: {success}/{len(tests)}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--transfer', type=str, nargs='+',
                        help='Use the suffix from a transfer attack')
    args = parser.parse_args()

    results = []
    if args.input_file:
        results.extend(process(args.input_file, transfer=args.transfer))
    if args.input_folder:
        for filename in os.listdir(args.input_folder):
            if not filename.endswith('.json'):
                continue
            results.extend(process(
                os.path.join(args.input_folder, filename),
                transfer=args.transfer))

    with open(args.output_file, 'w') as file:
        file.write(json.dumps(results, indent=2))

    print(f'{len(results)} examples saved')
