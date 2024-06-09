# Fooocus GPT2 Expansion
# Algorithm created by Lvmin Zhang at 2023, Stanford 
# Modified by power88 and GPT-4o for stable-diffusion-webui
# If used inside Fooocus, any use is permitted.
# If used outside Fooocus, only non-commercial use is permitted (CC-By NC 4.0).
# This applies to the word list, vocab, model, and algorithm.


import os
import torch
import math
import shutil
import gradio as gr
import psutil

from pathlib import Path
from modules.scripts import basedir
from huggingface_hub import hf_hub_download
from transformers.generation.logits_process import LogitsProcessorList
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from modules import scripts, paths_internal, errors, shared, script_callbacks
from modules.ui_components import InputAccordion


def text_encoder_device():
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    else:
        return torch.device("cpu")


def text_encoder_offload_device():
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    else:
        return torch.device("cpu")


def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = text_encoder_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024  # TODO
            mem_free_torch = mem_free_total
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch


# limitation of np.random.seed(), called from transformers.set_seed()
SEED_LIMIT_NUMPY = 2 ** 32
neg_inf = - 8192.0
ext_dir = Path(basedir())
fooocus_expansion_model_dir = Path(paths_internal.models_path) / "prompt_expansion"


def download_model():
    fooocus_expansion_model = fooocus_expansion_model_dir / "pytorch_model.bin"
    if not fooocus_expansion_model.exists():
        try:
            print(f'### webui-fooocus-prompt-expansion: Downloading model...')
            shutil.copytree(ext_dir / "models", fooocus_expansion_model_dir)
            hf_hub_download(repo_id='lllyasviel/misc', filename='fooocus_expansion.bin', local_dir=fooocus_expansion_model_dir)
            os.rename(fooocus_expansion_model_dir / 'fooocus_expansion.bin', fooocus_expansion_model)
        except Exception:
            errors.report('### webui-fooocus-prompt-expansion: Failed to download model', exc_info=True)
            print(f'Download the model manually from "https://huggingface.co/lllyasviel/misc/tree/main/fooocus_expansion.bin" and place it in {fooocus_expansion_model_dir}.')


def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace('  ', ' ')
    return x.strip(",. \r\n")


def remove_pattern(x, pattern):
    for p in pattern:
        x = x.replace(p, '')
    return x


def should_use_fp16(device=None, model_params=0, prioritize_performance=True):
    if device is not None:
        if hasattr(device, 'type'):
            if device.type == 'cpu':
                return False
        return False
    if torch.cuda.is_bf16_supported():
        return True
    props = torch.cuda.get_device_properties("cuda")
    if props.major < 6:
        return False

    fp16_works = False
    # FP16 is confirmed working on a 1080 (GP104) but it's a bit slower than FP32 so it should only be enabled
    # when the model doesn't actually fit on the card
    # TODO: actually test if GP106 and others have the same type of behavior
    nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050"]
    for x in nvidia_10_series:
        if x in props.name.lower():
            fp16_works = True

    if fp16_works:
        free_model_memory = (get_free_memory() * 0.9 - (1024 * 1024 * 1024))
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    # FP16 is just broken on these cards
    nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True


def is_device_mps(device):
    if hasattr(device, 'type'):
        if (device.type == 'mps'):
            return True
    return False


class FooocusExpansion:
    def __init__(self):
        global load_model_device
        download_model()
        print(f'Loading models from {fooocus_expansion_model_dir}')
        self.tokenizer = AutoTokenizer.from_pretrained(fooocus_expansion_model_dir)

        positive_words = open(os.path.join(fooocus_expansion_model_dir, 'positive.txt'),
                              encoding='utf-8').read().splitlines()
        positive_words = ['Ġ' + x.lower() for x in positive_words if x != '']

        self.logits_bias = torch.zeros((1, len(self.tokenizer.vocab)), dtype=torch.float32) + neg_inf

        debug_list = []
        for k, v in self.tokenizer.vocab.items():
            if k in positive_words:
                self.logits_bias[0, v] = 0
                debug_list.append(k[1:])

        print(f'Fooocus V2 Expansion: Vocab with {len(debug_list)} words.')

        self.model = AutoModelForCausalLM.from_pretrained(fooocus_expansion_model_dir)
        self.model.eval()

        load_model_device = text_encoder_device()
        offload_device = text_encoder_offload_device()

        # MPS hack
        if is_device_mps(load_model_device):
            load_model_device = torch.device('cpu')
            offload_device = torch.device('cpu')

        use_fp16 = should_use_fp16(device=load_model_device)

        if use_fp16:
            self.model.half()

        self.model.to(load_model_device)  # Ensure model is on the correct device

        print(f'Fooocus Expansion engine loaded for {load_model_device}, use_fp16 = {use_fp16}.')

    def unload_model(self):
        """Unload the model to free up memory."""
        del self.model
        torch.cuda.empty_cache()
        print('Model unloaded and memory cleared.')

    @torch.no_grad()
    @torch.inference_mode()
    def logits_processor(self, input_ids, scores):
        assert scores.ndim == 2 and scores.shape[0] == 1
        self.logits_bias = self.logits_bias.to(load_model_device)

        bias = self.logits_bias.clone().to(load_model_device)  # Ensure bias is on the correct device
        bias[0, input_ids[0].to(load_model_device).long()] = neg_inf  # Ensure input_ids are on the correct device
        bias[0, 11] = 0

        return scores + bias.to(scores.device)  # Ensure bias is on the same device as scores

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, prompt, seed):
        if prompt == '':
            return ''

        seed = int(seed) % SEED_LIMIT_NUMPY
        set_seed(seed)
        prompt = safe_str(prompt) + ','
        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data['input_ids'] = tokenized_kwargs.data['input_ids'].to(load_model_device)
        tokenized_kwargs.data['attention_mask'] = tokenized_kwargs.data['attention_mask'].to(load_model_device)

        current_token_length = int(tokenized_kwargs.data['input_ids'].shape[1])
        max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
        max_new_tokens = max_token_length - current_token_length

        features = self.model.generate(**tokenized_kwargs,
                                       top_k=100,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=True,
                                       logits_processor=LogitsProcessorList([self.logits_processor]))

        response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        result = safe_str(response[0])

        return result


def createPositive(positive, seed):
    try:
        expansion = FooocusExpansion()
        positive = expansion(positive, seed=seed)
        expansion.unload_model()  # Unload the model after use
        return positive
    except Exception as e:
        print(f"An error occurred: {str(e)}")


class FooocusPromptExpansion(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return 'Fooocus Prompt Expansion'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label="Fooocus Expansion") as is_enabled:
            seed = gr.Number(value=0, maximum=63, label="Seed", info="Seed for random number generator")
        return [is_enabled, seed]

    def process(self, p, is_enabled, seed):
        if not is_enabled:
            return

        for i, prompt in enumerate(p.all_prompts):
            positivePrompt = createPositive(prompt, seed)
            p.all_prompts[i] = positivePrompt

    def after_component(self, component, **kwargs):
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/7456#issuecomment-1414465888 helpfull link
        # Find the text2img textbox component
        if kwargs.get("elem_id") == "txt2img_prompt":  # postive prompt textbox
            self.boxx = component
        # Find the img2img textbox component
        if kwargs.get("elem_id") == "img2img_prompt":  # postive prompt textbox
            self.boxxIMG = component
