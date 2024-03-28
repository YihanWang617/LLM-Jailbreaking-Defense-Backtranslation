"""
Wrap targetLM with defense
The attacker has access to the defense strategy
"""
from packaging import version
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import fastchat
from fastchat.model import get_conversation_template
from fastchat.conversation import (Conversation, SeparatorStyle,
                                   register_conv_template, get_conv_template)
from language_models import GPT, Claude, PaLM, HuggingFace


def conv_template(template_name):
    if template_name == 'llama-2-new':
        # For compatibility with GCG
        template = get_conv_template(template_name)
    else:
        template = get_conversation_template(template_name)
    if template.name.startswith('llama-2'):
        template.sep2 = template.sep2.strip()
    return template


def load_indiv_model(model_name, max_memory=None):
    model_path, template = get_model_path_and_template(model_name)
    if model_name.startswith("gpt-"):
        lm = GPT(model_name)
    elif model_name in ["claude-2", "claude-instant-1"]:
        lm = Claude(model_name)
    elif model_name in ["palm-2"]:
        lm = PaLM(model_name)
    else:
        if max_memory is not None:
            max_memory = {i: f'{max_memory}MB'
                          for i in range(torch.cuda.device_count())}

        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True, device_map="auto",
                load_in_8bit=True,
                max_memory=max_memory).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        )
        if 'llama-2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)

    return lm, template


def get_model_path_and_template(model_name):
    full_model_dict={
        "gpt-4":{
            "path": "gpt-4",
            "template": "gpt-4"
        },
        "gpt-3.5-turbo": {
            "path": "gpt-3.5-turbo-0613",
            "template": "gpt-3.5-turbo"
        },
        "gpt-3.5-turbo-0301": {
            "path":"gpt-3.5-turbo-0301",
            "template":"gpt-3.5-turbo"
        },
        "gpt-3.5-turbo-0613": {
            "path":"gpt-3.5-turbo-0613",
            "template":"gpt-3.5-turbo"
        },
        "vicuna":{
            "path": "lmsys/vicuna-13b-v1.5",
            "template": "vicuna_v1.1"
        },
        "vicuna-13b-v1.5":{
            "path": 'lmsys/vicuna-13b-v1.5',
            "template": "vicuna_v1.1"
        },
        "llama-2":{
            "path": "meta-llama/Llama-2-13b-chat-hf",
            "template": "llama-2"
        },
        "llama-2-13b":{
            "path": "meta-llama/Llama-2-13b-chat-hf",
            "template": "llama-2"
        },
        "claude-instant-1":{
            "path": "claude-instant-1",
            "template": "claude-instant-1"
        },
        "claude-2":{
            "path": "claude-2",
            "template": "claude-2"
        },
        "palm-2":{
            "path": "palm-2",
            "template": "palm-2"
        },
    }

    path = full_model_dict[model_name]["path"]
    template = full_model_dict[model_name]["template"]

    if template == "llama-2":
        if version.parse(fastchat.__version__) < version.parse("0.2.24"):
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
                ),
                override=True
            )
            template = 'llama-2-new'
            print("Using LLaMA-2 with fastchat < 0.2.24. "
                  "Template changed to `llama-2-new`.")

    return path, template


class TargetLM():
    """
    Base class for target language models.

    Generates responses for prompts using a language model.
    The self.model attribute contains the underlying generation model.
    """
    def __init__(
        self,
        model_name: str,
        max_n_tokens: int,
        max_memory: int,
        temperature: float = 0.0,
        top_p: float = 1,
        preloaded_model: object = None,
        batch_size: int = 1,
        add_system_prompt: bool = True
    ):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        self.batch_size = batch_size
        self.add_system_prompt = add_system_prompt
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(
                model_name, max_memory=max_memory)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_response(self, prompts_list, responses_list=None, verbose=True):
        batchsize = len(prompts_list)
        convs_list = [conv_template(self.template)
                      for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                # Openai does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())

        if not self.add_system_prompt:
            print("Removing the system prompt. (Only for running PAP!)")
            if "gpt" in self.model_name:
                full_prompts = [[a for a in message if a['role'] != 'system']
                                for message in full_prompts]
            elif "llama" in self.model_name:
                for i, prompt in enumerate(full_prompts):
                    assert prompt.startswith('[INST]')
                    assert prompt.endswith('[/INST]')
                    # Just for consistency with PAP
                    # Do not use for other attacks
                    full_prompts[i] = (
                        self.model.tokenizer.bos_token +
                        "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]".format(
                            None, prompt[0][7:-8])
                    )
            else:
                raise NotImplementedError

        if verbose:
            print(f'Calling the TargetLM with {len(full_prompts)} prompts')
        outputs_list = []
        for i in tqdm(range((len(full_prompts)-1) // self.batch_size + 1),
                  desc="Target model inference on batch: ",
                  disable=(not verbose)):

            # Get the current batch of inputs
            batch = full_prompts[i * self.batch_size:(i+1) * self.batch_size]

            # Run a forward pass through the LLM for each perturbed copy
            batch_outputs = self.model.batched_generate(
                batch,
                max_n_tokens = self.max_n_tokens,
                temperature = self.temperature,
                top_p = self.top_p)

            outputs_list.extend(batch_outputs)
        return outputs_list

    def evaluate_log_likelihood(self, prompt, response):
        return self.model.evaluate_log_likelihood(prompt, response)


class DefensedTargetLM:
    def __init__(self, target_model, defense):
        self.target_model = target_model
        self.defense = defense

    def get_response(self, prompts_list, responses_list=None, verbose=True):
        if responses_list is not None:
            assert len(responses_list) == len(prompts_list)
        else:
            responses_list = [None for _ in prompts_list]
        defensed_response = [
            self.defense.defense(
                prompt, self.target_model, response=response, verbose=verbose
            )
            for prompt, response in zip(prompts_list, responses_list)]
        return defensed_response

    def evaluate_log_likelihood(self, prompt, response):
        return self.target_model.evaluate_log_likelihood(prompt, response)
