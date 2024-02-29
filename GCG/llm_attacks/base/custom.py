"""Custom code modifying GCG."""

import time
from copy import deepcopy
import queue
import torch
from transformers import AutoModelForCausalLM


class ModelWorkerCustom(object):
    """No multiprocessing."""

    def __init__(self, model_path, model_kwargs, tokenizer, conv_template,
                 device, use_8bit=True, batch_size=1024):
        print('Using ModelWorkerCustom, '
              f'use_8bit={use_8bit}, batch_size={batch_size}')
        if use_8bit:
            model_kwargs['load_in_8bit'] = True
        else:
            model_kwargs['torch_dtype'] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map='auto',
            **model_kwargs
        ).eval()
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.tasks = queue.Queue()
        self.results = queue.Queue()
        self.batch_size = batch_size
        self.process = None

    def start(self):
        pass

    def stop(self):
        torch.cuda.empty_cache()
        return self

    def __call__(self, ob, fn, *args, **kwargs):
        ob = deepcopy(ob)
        if fn == "grad":
            stime = time.time()
            with torch.enable_grad():
                self.results.put(ob.grad(*args, **kwargs))
            print('Time for grad call:', time.time() - stime)
        else:
            with torch.no_grad():
                if fn == "logits":
                    stime = time.time()
                    assert len(args) == 2
                    model, cand = args
                    outputs = []
                    loss_slice = slice(ob._target_slice.start-1, ob._target_slice.stop-1)
                    for i in range((len(cand) + self.batch_size - 1) // self.batch_size):
                        logits, ids = ob.logits(
                            model,
                            cand[i * self.batch_size : (i + 1) * self.batch_size],
                            **kwargs)
                        logits = logits[:, loss_slice, :]
                        ids = ids[:, ob._target_slice]
                        outputs.append((logits, ids))
                    output = (torch.concat([item[0] for item in outputs]),
                              torch.concat([item[1] for item in outputs]))
                    self.results.put(output)
                    print('Time for logits call:', time.time() - stime)
                elif fn == "contrast_logits":
                    self.results.put(ob.contrast_logits(*args, **kwargs))
                elif fn == "test":
                    self.results.put(ob.test(*args, **kwargs))
                elif fn == "test_loss":
                    self.results.put(ob.test_loss(*args, **kwargs))
                else:
                    self.results.put(fn(*args, **kwargs))
