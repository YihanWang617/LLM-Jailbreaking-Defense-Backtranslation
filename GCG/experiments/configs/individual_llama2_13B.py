import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():

    config = default_config()

    config.tokenizer_paths = ["meta-llama/Llama-2-13b-chat-hf"]
    config.model_paths = ["meta-llama/Llama-2-13b-chat-hf"]
    config.conversation_templates = ['llama-2-new']
    config.n_steps = 200
    config.test_steps = 5
    config.batch_size = config.model_batch_size = 512

    return config
