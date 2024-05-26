import os
import pathlib
import shutil
from huggingface_hub import hf_hub_download
from modules.scripts import basedir

ext_dir = basedir()
fooocus_expansion_path = pathlib.Path(ext_dir) / "models" / "prompt_expansion"
base_model_path = pathlib.Path(ext_dir) / "extensions" / "webui-fooocus-prompt-expansion" / "models"


if not os.path.exists(os.path.join(fooocus_expansion_path, 'pytorch_model.bin')):
    try:
        print(f'### webui-fooocus-prompt-expansion: Downloading model...')
        shutil.copytree(os.path.join(base_model_path), fooocus_expansion_path)
        hf_hub_download(repo_id='lllyasviel/misc', filename='fooocus_expansion.bin', local_dir=os.path.join(fooocus_expansion_path), resume_download=True, local_dir_use_symlinks=False)
        os.rename(os.path.join(fooocus_expansion_path, 'fooocus_expansion.bin'), os.path.join(fooocus_expansion_path, 'pytorch_model.bin'))
    except Exception as e:
        print(f'### webui-fooocus-prompt-expansion: Failed to download model...')
        print(e)
        print(f'### webui-fooocus-prompt-expansion: To enable this custom node, please download the model manually from "https://huggingface.co/lllyasviel/misc/tree/main/fooocus_expansion.bin" and place it in {fooocus_expansion_path}.')
else:
    pass
