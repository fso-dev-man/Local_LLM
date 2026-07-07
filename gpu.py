from diffusers import StableDiffusionXLPipeline
import torch

# 1. Load the pipeline directly to GPU using float16 for speed and VRAM savings
pipe = StableDiffusionXLPipeline.from_single_file(
    r"C:\SDXL\sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

# 2. Add GPU optimizations to prevent Out-Of-Memory (OOM) errors
pipe.enable_model_cpu_offload()  # Smartly moves parts of the model to CPU when not in use
pipe.enable_vae_tiling()         # Saves VRAM during the high-resolution image decoding step

prompt = "now dog and cat are fighting each other this is the animal war, draw the picture as a modern style"

# 3. Generate the image using a GPU-supported random seed generator
generator = torch.Generator("cuda").manual_seed(42)

image = pipe(
    prompt,
    num_inference_steps=20,
    height=1024,
    width=1024,
    generator=generator
).images[0]

image.save("result.png")