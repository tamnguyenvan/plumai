import os
import io

from modal import (
    App,
    Image,
    build,
    enter,
    gpu,
    method,
    web_endpoint
)


flux_image = (
    Image.from_registry(
        "nvidia/cuda:12.3.1-base-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1", "aria2"
    )
    .run_commands("git clone -b totoro3 https://github.com/camenduru/ComfyUI /TotoroUI")
    .pip_install(
        "torchsde",
        "einops",
        "diffusers",
        "accelerate",
        "xformers",
        "torchvision",
        "transformers"
    )
)


app = App("flux.1-schnell")

with flux_image.imports():
    import random
    import torch
    import numpy as np
    from PIL import Image

gpu_type = os.getenv("GPU_TYPE", "A10G")
@app.cls(gpu=gpu_type, container_idle_timeout=240, image=flux_image)
class Model:
    @build()
    def build(self):
        import os
        import subprocess

        os.chdir("/TotoroUI")

        subprocess.run("echo && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors -d /TotoroUI/models/unet -o flux1-schnell.safetensors", shell=True,)
        subprocess.run("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft -d /TotoroUI/models/vae -o ae.sft", shell=True,)
        subprocess.run("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors -d /TotoroUI/models/clip -o clip_l.safetensors", shell=True,)
        subprocess.run("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors -d /TotoroUI/models/clip -o t5xxl_fp8_e4m3fn.safetensors", shell=True,)

    @enter()
    def enter(self):
        import os
        import sys

        project_path = "/TotoroUI"
        if project_path not in sys.path:
            sys.path.append(project_path)

        os.chdir(project_path)

        import torch
        from nodes import NODE_CLASS_MAPPINGS
        from totoro_extras import nodes_custom_sampler

        self.DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        self.UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        self.RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
        self.BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
        self.KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        self.BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
        self.SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        self.VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
        self.VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        self.EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        with torch.inference_mode():
            self.clip = self.DualCLIPLoader.load_clip("t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux")[0]
            self.unet = self.UNETLoader.load_unet("flux1-schnell.safetensors", "fp8_e4m3fn")[0]
            self.vae = self.VAELoader.load_vae("ae.sft")[0]

    def _inference(self, prompt, width, height, seed, guidance_scale, n_steps):
        import sys

        project_path = "/TotoroUI"
        if project_path not in sys.path:
            sys.path.append(project_path)

        from totoro import model_management

        with torch.inference_mode():
            sampler_name = "euler"
            scheduler = "simple"

            if seed < 0:
                seed = random.randint(0, 18446744073709551615)

            cond, pooled = self.clip.encode_from_tokens(self.clip.tokenize(prompt), return_pooled=True)
            cond = [[cond, {"pooled_output": pooled}]]
            noise = self.RandomNoise.get_noise(seed)[0]
            guider = self.BasicGuider.get_guider(self.unet, cond)[0]
            sampler = self.KSamplerSelect.get_sampler(sampler_name)[0]
            sigmas = self.BasicScheduler.get_sigmas(self.unet, scheduler, n_steps, 1.0)[0]
            latent_image = self.EmptyLatentImage.generate(closestNumber(width, 16), closestNumber(height, 16))[0]
            sample, sample_denoised = self.SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
            model_management.soft_empty_cache()
            decoded = self.VAEDecode.decode(self.vae, sample)[0].detach()
            image = Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0])

            byte_stream = io.BytesIO()
            image.save(byte_stream, format="JPEG")
            return byte_stream

    @method()
    def inference(self, prompt, width, height, seed, guidance_scale, n_steps):
        return self._inference(prompt, width, height, seed, guidance_scale, n_steps).getvalue()

    @web_endpoint(docs=True)
    def web_inference(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        guidance_scale: float = 3.5,
        n_steps: int = 20
    ):
        from fastapi import Response
        return Response(
            content=self._inference(
                prompt,
                width,
                height,
                seed,
                guidance_scale,
                n_steps
            ).getvalue(),
            media_type="image/jpeg",
        )


def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2
