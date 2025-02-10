import warnings
warnings.filterwarnings("ignore")

import numpy as np
import gradio as gr
from pathlib import Path
from omegaconf import OmegaConf
from sampler_invsr import InvSamplerSR

from utils import util_common
from utils import util_image
from basicsr.utils.download_util import load_file_from_url

def get_configs(num_steps=1, chopping_size=128, seed=12345):
    configs = OmegaConf.load("./configs/sample-sd-turbo.yaml")

    if num_steps == 1:
        configs.timesteps = [200,]
    elif num_steps == 2:
        configs.timesteps = [200, 100]
    elif num_steps == 3:
        configs.timesteps = [200, 100, 50]
    elif num_steps == 4:
        configs.timesteps = [200, 150, 100, 50]
    elif num_steps == 5:
        configs.timesteps = [250, 200, 150, 100, 50]
    else:
        assert num_steps <= 250
        configs.timesteps = np.linspace(
            start=250, stop=0, num=num_steps, endpoint=False, dtype=np.int64()
        ).tolist()
    print(f'Setting timesteps for inference: {configs.timesteps}')

    configs.sd_path = "./weights"
    util_common.mkdir(configs.sd_path, delete=False, parents=True)
    configs.sd_pipe.params.cache_dir = configs.sd_path

    started_ckpt_name = "noise_predictor_sd_turbo_v5.pth"
    started_ckpt_dir = "./weights"
    util_common.mkdir(started_ckpt_dir, delete=False, parents=True)
    started_ckpt_path = Path(started_ckpt_dir) / started_ckpt_name
    if not started_ckpt_path.exists():
        load_file_from_url(
            url="https://huggingface.co/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth",
            model_dir=started_ckpt_dir,
            progress=True,
            file_name=started_ckpt_name,
        )
    configs.model_start.ckpt_path = str(started_ckpt_path)

    configs.bs = 1
    configs.seed = seed
    configs.basesr.chopping.pch_size = chopping_size
    configs.basesr.chopping.extra_bs = 4

    return configs

def predict_single(in_path, num_steps=1, chopping_size=128, seed=12345):
    configs = get_configs(num_steps=num_steps, chopping_size=chopping_size, seed=seed)
    sampler = InvSamplerSR(configs)

    out_dir = Path('invsr_output')
    if not out_dir.exists():
        out_dir.mkdir()
    sampler.inference(in_path, out_path=out_dir, bs=1)

    out_path = out_dir / f"{Path(in_path).stem}.png"
    assert out_path.exists(), 'Super-resolution failed!'
    im_sr = util_image.imread(out_path, chn="rgb", dtype="uint8")

    return im_sr, str(out_path)

title = "InvSR App (AI Temple)"

description = r"""
🔥 InvSR is an image super-resolution method via Diffusion Inversion, supporting arbitrary sampling steps.<br>
Made by: <a href='https://patreon.com/AITemple' target='_blank'><b>https://patreon.com/AITemple</b></a><br>
"""

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Tabs():
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="filepath", label="Input: Low Quality Image")
                    num_steps = gr.Dropdown(
                        choices=[1,2,3,4,5],
                        value=1,
                        label="Number of steps",
                    )
                    chopping_size = gr.Dropdown(
                        choices=[128, 256],
                        value=128,
                        label="Chopping size",
                    )
                    seed = gr.Number(value=12345, precision=0, label="Random seed")
                    process_btn = gr.Button("Process")

                with gr.Column():
                    output_image = gr.Image(type="numpy", label="Output: High Quality Image")
                    output_file = gr.File(label="Download the output")

            process_btn.click(
                fn=predict_single,
                inputs=[input_image, num_steps, chopping_size, seed],
                outputs=[output_image, output_file]
            )

demo.queue(max_size=5)
demo.launch(share=False, server_name="0.0.0.0")
