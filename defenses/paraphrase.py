from judge import check_rejection_harmful
from .defensebase import DefenseBase


def parse_output(raw_output):
    paraphrase_prompt = raw_output.strip().strip(']').strip('[')
    return paraphrase_prompt


def paraphrase(prompt, lm, verbose=True):
    _prompt = f'paraphrase the following paragraph: \n"{prompt}"\n\n'


    if verbose:
        print('Asking the model to paraphrase the prompt:')
        print(_prompt)

    output = lm.get_response([_prompt], verbose=verbose)[0]

    paraphrase_prompt = parse_output(output)

    if verbose:
        print('Paraphrased prompt:', paraphrase_prompt)
    return paraphrase_prompt


class ParaphraseDefense(DefenseBase):
    def __init__(self, defense_method, paraphrase_lm):
        super(ParaphraseDefense, self).__init__(defense_method)
        self.paraphrase_lm = paraphrase_lm

    def defense(self, prompt, target_lm, response=None, verbose=False):
        paraphrase_prompt = paraphrase(prompt, self.paraphrase_lm)
        if paraphrase_prompt is None:
            if response is None:
                response = target_lm.get_response([prompt], verbose=verbose)[0]
            return response
        else:
            if "\n" in paraphrase_prompt:
                paraphrase_prompt = "\n".join(paraphrase_prompt.split('\n')[1:]) # remove useless parts.
            paraphrase_response = target_lm.get_response([paraphrase_prompt], verbose=verbose)[0]
            return paraphrase_response
