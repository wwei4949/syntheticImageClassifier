from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
import os
from pathlib import Path
from PIL import Image

from helper_func import color_to_binary_mask


if __name__ == "__main__":
    model_dir = "models"
    results_dir = "results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    num_img_per_prompt = 1
    damage_to_mask_color = {  # mapping from damage type to RGB mask color
        "corner_impact": (163, 73, 164),  # purple
        "windshield_damage": (63, 72, 204),  # dark blue
        "side_impact": (0, 162, 232)  # light blue
    }

    print("image generation")
    generation_model = f"{model_dir}/stable-diffusion-v1-5"
    if not os.path.exists(generation_model):
        raise Exception(f"generation model does not exist: {generation_model}")
    generation_pipe = StableDiffusionPipeline.from_pretrained(f"{model_dir}/stable-diffusion-v1-5")
    generation_pipe = generation_pipe.to("cuda")
    prompts = ["taxi cab with broken bumper"]
    for prompt in prompts:
        for i in range(0, num_img_per_prompt):
            print(f"generating from prompt '{prompt}' [{i+1}/{num_img_per_prompt}]")
            image = generation_pipe(prompt).images[0]
            image.save(f"{results_dir}/{prompt.replace(' ', '-')}_{i}.png")

    del generation_pipe  # clear vmem for inpainting

    print("image inpainting")
    inpainting_model = f"{model_dir}/stable-diffusion-inpainting"
    if not os.path.exists(inpainting_model):
        raise Exception(f"inpainting model does not exist: {inpainting_model}")
    inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(f"{model_dir}/stable-diffusion-inpainting")
    inpainting_pipe = inpainting_pipe.to("cuda")
    input_image = Image.open("examples/car.jpg")
    input_mask = color_to_binary_mask(img_path="examples/car_mask.png",
                                      mask_color=damage_to_mask_color["corner_impact"])
    prompts = ["car crash"]
    for prompt in prompts:
        for i in range(0, num_img_per_prompt):
            print(f"inpainting from prompt '{prompt}' [{i+1}/{num_img_per_prompt}]")
            result = inpainting_pipe(prompt=prompt, image=input_image, mask_image=input_mask).images[0]
            result.save(f"{results_dir}/{prompt.replace(' ', '-')}_{i}.png")
