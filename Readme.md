# SDXL Local Environment Setup Guide (Windows)

This guide explains how to set up a local Python environment, install all required dependencies, download the SDXL model, and generate images using either a CPU or an NVIDIA GPU.

---

# Step 1. Install Python

Download and install Python 3.12 (or the version compatible with your project).

During installation, **make sure to enable**:

* ✅ Add Python to PATH
* ✅ Install pip

After installation, verify that Python and pip are available.

```cmd
python --version
pip --version
```

If both commands display version information, the installation was successful.

---

# Step 2. Create a Python Virtual Environment

Navigate to your project directory.

```cmd
cd C:\SDXL
```

Create a virtual environment.

```cmd
python -m venv venv
```

Activate the virtual environment.

**Windows Command Prompt**

```cmd
venv\Scripts\activate
```

**Windows PowerShell**

If PowerShell blocks script execution, temporarily allow it for the current session.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment.

```powershell
.\venv\Scripts\Activate.ps1
```

After activation, your terminal prompt should look similar to:

```text
(venv) C:\SDXL>
```

All Python packages installed from this point forward will be isolated inside the virtual environment.

---

# Step 3. Install Required Python Packages

There are two installation methods.

---

## Method 1 (Recommended): Install from Pre-downloaded Wheel Files

If you already have a compressed archive (for example, **whl.zip**) containing all required `.whl` files, this is the fastest method and works completely offline.

### 1. Extract the archive

Extract

```
whl.zip
```

to

```
C:\SDXL\SDXL_Wheels
```

The directory should contain all downloaded wheel files.

Example:

```
C:\SDXL
│
├── SDXL_Wheels
│     accelerate-....
│     diffusers-....
│     numpy-....
│     pillow-....
│     torch-....
│     torchvision-....
│     torchaudio-....
│     ...
│
├── requirements.txt
└── venv
```

### 2. Install all dependencies

```cmd
pip install --no-index --find-links=C:\SDXL\SDXL_Wheels -r requirements.txt
```

Explanation:

* `--no-index` prevents pip from accessing the internet.
* `--find-links` tells pip where to locate the local wheel files.
* `requirements.txt` specifies the packages to install.

---

## Method 2: Download All Dependencies from the Internet

If you do not already have the wheel files, download them first.

### PowerShell Users

If necessary, allow script execution temporarily.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Download all required packages

```cmd
pip download -r requirements.txt -d C:\SDXL\SDXL_Wheels
```

This command downloads every package listed in `requirements.txt` together with all required dependencies.

### Download CPU-only PyTorch packages (if required)

```cmd
pip download torch==2.12.1+cpu torchvision==0.27.1+cpu torchaudio==2.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu -d C:\SDXL\SDXL_Wheels
```

After downloading completes, install the packages locally.

```cmd
pip install --no-index --find-links=C:\SDXL\SDXL_Wheels -r requirements.txt
```

---

# Step 4. Verify the Installation

Confirm that the required packages are installed.

```cmd
pip list
```

or

```cmd
pip freeze
```

You should see packages such as:

* torch
* torchvision
* torchaudio
* diffusers
* transformers
* accelerate
* safetensors
* huggingface_hub

---

# Step 5. Download the SDXL Model

Use the Hugging Face CLI to download the SDXL Base model.

```cmd
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir ./stable-diffusion-xl-base-1.0
```

After downloading, the directory structure should resemble:

```
C:\SDXL
│
├── stable-diffusion-xl-base-1.0
│      model_index.json
│      scheduler
│      tokenizer
│      tokenizer_2
│      text_encoder
│      text_encoder_2
│      unet
│      vae
│      ...
```

If you instead have the single-file checkpoint (`sd_xl_base_1.0.safetensors`), place it somewhere convenient, for example:

```
C:\SDXL\sd_xl_base_1.0.safetensors
```

---

# Step 6. Run SDXL on CPU

The following script loads the SDXL checkpoint entirely on the CPU.

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_single_file(
    r"C:\SDXL\sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float32,
    use_safetensors=True
)

pipe = pipe.to("cpu")

pipe.enable_attention_slicing()

prompt = (
    "now dog and cat are fighting each other "
    "this is the animal war, draw the picture as a modern style"
)

image = pipe(
    prompt,
    num_inference_steps=20,
    height=1024,
    width=1024
).images[0]

image.save("result.png")
```

The generated image will be saved as:

```
result.png
```

---

# Step 7. Run SDXL on an NVIDIA GPU

If CUDA is available, the following script provides significantly faster inference.

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_single_file(
    r"C:\SDXL\sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe = pipe.to("cuda")

pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = (
    "now dog and cat are fighting each other "
    "this is the animal war, draw the picture as a modern style"
)

generator = torch.Generator("cuda").manual_seed(42)

image = pipe(
    prompt,
    num_inference_steps=20,
    height=1024,
    width=1024,
    generator=generator
).images[0]

image.save("result.png")
```

This version uses:

* `float16` precision to reduce GPU memory usage
* `enable_model_cpu_offload()` to minimize out-of-memory errors
* `enable_vae_tiling()` to reduce VRAM usage during image decoding
* A fixed random seed (`42`) for reproducible image generation

---

# Expected Project Structure

```
C:\SDXL
│
├── venv
├── SDXL_Wheels
├── stable-diffusion-xl-base-1.0
├── requirements.txt
├── sd_xl_base_1.0.safetensors
├── cpu_generate.py
├── gpu_generate.py
└── result.png
```

Once all dependencies are installed and the model files are downloaded, you can activate the virtual environment and execute either the CPU or GPU script to generate SDXL images locally.