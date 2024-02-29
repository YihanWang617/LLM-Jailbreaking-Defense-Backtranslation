
from . import common
from .config import ATTACK_TEMP, ATTACK_TOP_P
from tqdm import tqdm
import sys
sys.path.append("..")
from jailbreakingDefense.utils import load_target_model
from .models import load_indiv_model


def load_attack_and_target_models(args):
    # Load attack model and tokenizer
    attackLM = AttackLM(model_name = args.attack_model,
                        max_n_tokens = args.attack_max_n_tokens,
                        max_n_attack_attempts = args.max_n_attack_attempts,
                        temperature = ATTACK_TEMP, # init to 1
                        top_p = ATTACK_TOP_P, # init to 0.9
                        max_memory = args.max_memory,
                        )
    targetLM = load_target_model(args, args.target_model, args.target_max_n_tokens,
                                 args.max_memory, attackLM.model, defense_method=args.defense_method)
    return attackLM, targetLM


class AttackLM():
    """
        Base class for attacker language models.

        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self,
                model_name: str,
                max_n_tokens: int,
                max_n_attack_attempts: int,
                temperature: float,
                top_p: float,
                max_memory: int):

        self.batch_size = 2
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(
            model_name, max_memory=max_memory)

        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()

    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model.
        Only valid outputs in proper JSON format are returned. If an output isn't generated
        successfully after max_n_attack_attempts, it's returned as None.

        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.

        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """

        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \""""

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])

        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs
            outputs_list = []
            print(f'Calling the AttackLM with {len(full_prompts_subset)} prompts')
            for i in tqdm(range((len(full_prompts_subset)-1) // self.batch_size + 1)):
                # Get the current batch of inputs
                batch = full_prompts_subset[i * self.batch_size:(i+1) * self.batch_size]

                # Run a forward pass through the LLM for each perturbed copy
                batch_outputs = self.model.batched_generate(batch, 
                                                            max_n_tokens = self.max_n_tokens,
                                                            temperature = self.temperature,
                                                            top_p = self.top_p)

                outputs_list.extend(batch_outputs)

            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "gpt" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output)

                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(json_str)  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)

            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate

            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if len([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")

        return valid_outputs

