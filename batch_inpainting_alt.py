from diffusers import StableDiffusionInpaintPipeline
import json
import os
from pathlib import Path
from PIL import Image
import random

from helper_func import color_to_binary_mask


if __name__ == "__main__":
    image_data_dir = "image_data"
    model_dir = "models"
    results_dir = "results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    damage_to_mask_color = {  # mapping from damage type to RGB mask color
        "corner_impact": (163, 73, 164),  # purple
        "windshield_damage": (63, 72, 204),  # dark blue
        "side_impact": (0, 162, 232)  # light blue
    }

    adjectives = [
        "acutely", "awfully", "exceedingly", "exceptionally", "excessively", "extraordinarily", "highly", "hugely",
        "immensely", "intensely", "overly", "quite", "remarkably", "severely", "strikingly", "terribly", "terrifically",
        "totally", "utterly", "very", "slightly"
    ]
    broken_synonyms = [
        "busted", "collapsed", "cracked", "crippled", "crumbled", "crushed", "damaged", "defective", "demolished",
        "fractured", "fragmented", "mangled", "mutilated", "ruptured", "severed", "shattered", "smashed", "burst",
        "disintegrated", "dismembered", "pulverized", "separated", "shredded", "split", "fragmented", "in pieces"
    ]

    mask_pngs = [f.as_posix() for f in Path(image_data_dir).rglob("*.png")]
    print(f"found {len(mask_pngs)} mask pngs in {image_data_dir}")
    img_jpgs = [f.replace(".png", ".jpg").replace("masked", "image_data") for f in mask_pngs]
    for img_jpg, mask_png in zip(img_jpgs, mask_pngs):
        if not os.path.exists(img_jpg):
            raise Exception(f"jpg associated to {mask_png} not found: {img_jpg}")

    inpainting_prompts = []
    num_prompts_per_unique_image_mask = 5
    for i in range(0, num_prompts_per_unique_image_mask):
        for img_jpg, mask_png in zip(img_jpgs, mask_pngs):
            for vehicle_part, damage_type in zip(("glass", "bumper", "vehicle"),
                                                 ("windshield_damage", "corner_impact", "side_impact")):
                prompt = f"{random.choice(adjectives)} {random.choice(broken_synonyms)} {vehicle_part}"
                input_mask = color_to_binary_mask(img_path=mask_png, mask_color=damage_to_mask_color[damage_type])
                if input_mask:  # some images don't have all types of damage
                    # could save input_mask to prevent reloading the image, but better to keep this json-parsable
                    inpainting_prompts.append({"img_jpg": img_jpg, "mask_png": mask_png,
                                               "damage_type": damage_type, "prompt": prompt})

    # add an index to keep track of outputs
    for idx, prompt in enumerate(inpainting_prompts, start=0):
        prompt["identifier"] = idx

    with open("inpainting_prompts.json", "w") as f:
        json.dump(inpainting_prompts, f, indent=4)

    # inpainting_model = f"{model_dir}/stable-diffusion-inpainting"
    inpainting_model = f"{model_dir}/stable-diffusion-2-inpainting"
    if not os.path.exists(inpainting_model):
        raise Exception(f"inpainting model does not exist: {inpainting_model}")
    inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(inpainting_model)
    inpainting_pipe = inpainting_pipe.to("cuda")

    for i, inpainting_prompt in enumerate(inpainting_prompts, start=0):
        identifier = inpainting_prompt["identifier"]
        prompt = inpainting_prompt["prompt"]
        input_image = Image.open(inpainting_prompt["img_jpg"])
        input_mask = color_to_binary_mask(img_path=inpainting_prompt["mask_png"],
                                          mask_color=damage_to_mask_color[inpainting_prompt["damage_type"]])
        print(f"inpainting from prompt '{prompt}' [{i+1}/{len(inpainting_prompts)}]")
        result = inpainting_pipe(prompt=prompt, image=input_image, mask_image=input_mask).images[0]
        result.save(f"{results_dir}/{str(identifier).zfill(3)}_{prompt.replace(' ', '_')}.png")
