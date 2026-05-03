from diffusers import StableDiffusionPipeline
import torch

# Load model (first time will download ~4GB)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Prompt
prompt = "A CCTV scene showing a person running on a street, realistic, surveillance style"

# Generate image
image = pipe(prompt).images[0]

# Save output
image.save("outputs/generated_scene.png")

print("✅ Image generated successfully!")