from types import SimpleNamespace
from judge import MatchingJudge
from .backtranslation import BackTranslationDefense
from .self_check import SelfCheckResponseDefense
from .smoothllm import SmoothLLMDefense
from .paraphrase import ParaphraseDefense
from .defensebase import DefenseBase
from models import TargetLM

def load_defense(args, defense_method, preloaded_model=None):
    if defense_method == 'smoothLLM':
        args = SimpleNamespace(judge_model='matching', target_str="", goal="")
        judge = MatchingJudge(args)
        return SmoothLLMDefense(defense_method, judge)
    elif defense_method.startswith("backtranslation"):
        if 'threshold' in defense_method:
            threshold = float(defense_method.split('_')[-1])
            defense_method = "backtranslation"
            if threshold > 0:
                threshold = -threshold
            args.backtranslation_threshold = threshold
        else:
            args.backtranslation_threshold = -2.0
        print(f'Using threshold {args.backtranslation_threshold} for backtranslation')
        infer_lm = TargetLM(model_name=args.backtranslation_infer_model,
                            max_n_tokens=args.target_max_n_tokens,
                            max_memory=args.max_memory,
                            preloaded_model=preloaded_model)
        return BackTranslationDefense(
            defense_method, infer_lm, args.return_new_response_anyway,
            threshold=args.backtranslation_threshold)
    elif defense_method == 'self_check_response':
        return SelfCheckResponseDefense(defense_method, args.self_check_threshold)
    elif defense_method == 'paraphrase_prompt':
        paraphrase_lm = TargetLM(model_name=args.paraphrase_model,
                                 max_n_tokens=1024,
                                 max_memory=args.max_memory,
                                 preloaded_model=None,
                                 add_system_prompt=not args.no_system_prompt)
        return ParaphraseDefense(defense_method, paraphrase_lm)
    elif defense_method is None or defense_method == 'None':
        return DefenseBase(defense_method)
    raise NotImplementedError()
