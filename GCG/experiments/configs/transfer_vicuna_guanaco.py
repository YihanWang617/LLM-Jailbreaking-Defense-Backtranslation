import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():

    config = default_config()

    config.transfer = True
    config.logfile = ""
    config.evalfile = ""

    config.progressive_goals = True
    config.stop_on_success = True
    config.tokenizer_paths = [
        "TheBloke/guanaco-7B-HF",
        "TheBloke/guanaco-13B-HF",
        "lmsys/vicuna-7b-v1.5",
        "lmsys/vicuna-13b-v1.5"
    ]
    config.tokenizer_kwargs = [{"use_fast": False}, {"use_fast": False}, {"use_fast": False}, {"use_fast": False}]
    config.model_paths = [
        "TheBloke/guanaco-7B-HF",
        "TheBloke/guanaco-13B-HF",
        "lmsys/vicuna-7b-v1.5",
        "lmsys/vicuna-13b-v1.5"
    ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False},
        {"low_cpu_mem_usage": True, "use_cache": False},
        {"low_cpu_mem_usage": True, "use_cache": False},
        {"low_cpu_mem_usage": True, "use_cache": False}
    ]
    config.conversation_templates = ["guanaco", "guanaco", "vicuna_v1.1", "vicuna_v1.1"]
    config.devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    config.n_train_data = 25
    config.n_test_data = 25
    config.n_steps = 500
    config.test_steps = 1
    config.batch_size = 512
    config.model_batch_size = 256
    config.num_train_models = 4

    return config
