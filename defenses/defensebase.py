class DefenseBase():
    def __init__(self, defense_method):
        self.defense_method = defense_method

    def defense(self, prompt, target_lm, response=None, verbose=False):
        if response is None:
            response = target_lm.get_response([prompt], verbose=verbose)[0]
        return response
