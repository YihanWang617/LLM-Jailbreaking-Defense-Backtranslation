import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    config = default_config()

    config.tokenizer_paths = ["lmsys/vicuna-13b-v1.5"]
    config.model_paths = ["lmsys/vicuna-13b-v1.5"]
    config.conversation_templates = ['vicuna_v1.1']
    config.n_steps = 200
    config.test_steps = 5
    config.batch_size = config.model_batch_size = 512

    return config
