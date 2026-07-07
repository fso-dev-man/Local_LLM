from diffusers import StableDiffusionXLPipeline
import torch

# 1. Load the pipeline to CPU directly
pipe = StableDiffusionXLPipeline.from_single_file(
    r"C:\SDXL\sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float32,
    use_safetensors=True
)
pipe = pipe.to("cpu")

# 2. Add CPU optimization to save RAM
pipe.enable_attention_slicing()

prompt = "now dog and cat are fighting each other this is the animal war, draw the picture as a modern style"

# 3. Generate the image
image = pipe(
    prompt,
    num_inference_steps=20,
    height=1024,
    width=1024
).images[0]

image.save("result.png")