from ml_collections import config_dict
from fastchat.conversation import Conversation, register_conv_template, SeparatorStyle


# Running GCG may require fastchat == 0.2.20
# But here we update the system prompt of LLaMA-2 to match the latest fastchat
llama_2_template = 'llama-2-new'
print('Using fastchat 0.2.20. Removing the system prompt for LLaMA-2.')
register_conv_template(
    Conversation(
        name="llama-2-new",
        system="<s>[INST] <<SYS>>\n\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
        stop_token_ids=[2],
    )
)


def get_config():
    config = config_dict.ConfigDict()

    # Experiment type
    config.transfer = False

    # General parameters
    config.target_weight=1.0
    config.control_weight=0.0
    config.progressive_goals=False
    config.progressive_models=False
    config.anneal=False
    config.incr_control=False
    config.stop_on_success=False
    config.verbose=True
    config.allow_non_ascii=False
    config.num_train_models=1

    # Results
    config.result_prefix = 'results/individual_vicuna7b'

    # tokenizers
    config.tokenizer_paths=['lmsys/vicuna-7b-v1.3']
    config.tokenizer_kwargs=[{"use_fast": False}]

    config.model_paths=['lmsys/vicuna-7b-v1.3']
    config.model_kwargs=[{"low_cpu_mem_usage": True, "use_cache": False}]
    config.conversation_templates=['vicuna_v1.1']
    config.devices=['cuda']

    # data
    config.train_data = ''
    config.test_data = ''
    config.n_train_data = 50
    config.n_test_data = 0
    config.data_offset = 0

    # attack-related parameters
    config.attack = 'gcg'
    config.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    config.n_steps = 500
    config.test_steps = 50
    config.batch_size = 512
    config.lr = 0.01
    config.topk = 256
    config.temp = 1
    config.filter_cand = True

    config.gbda_deterministic = True

    # custom options
    config.model_batch_size = 1024
    config.use_8bit = False

    return config
