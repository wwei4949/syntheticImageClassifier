

from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
import os
from pathlib import Path
from PIL import Image

from helper_func import color_to_binary_mask
import glob
from typing import List, Dict

damage_to_mask_color = {  # mapping from damage type to RGB mask color
        "corner_impact": (163, 73, 164),  # purple
        "windshield_damage": (63, 72, 204),  # dark blue
        "side_impact": (0, 162, 232)  # light blue
    }

model_dir = "models"
results_dir = "results"
Path(results_dir).mkdir(parents=True, exist_ok=True)

def run_inpainting(image_file: str,
                   masked_file: str,
                   prompts: List[str],
                   pipe_params: Dict[str,str] = {}):
    input_image = Image.open(image_file)
    
    base_name = str(Path(image_file).stem)
    color_keys = damage_to_mask_color.keys()
    for prompt, col in zip(prompts, color_keys):
        input_mask = color_to_binary_mask(img_path=masked_file, mask_color=damage_to_mask_color[col])
        for i in range(0, num_img_per_prompt):
            print(f"inpainting from prompt '{prompt}' [{i+1}/{num_img_per_prompt}] with extra params: {pipe_params}")
            result = inpainting_pipe(prompt=prompt, image=input_image, mask_image=input_mask, **pipe_params).images[0]
            result.save(f"{results_dir}/{base_name}_{prompt.replace(' ', '-')}_{i}.png")

    
if __name__ == "__main__":
    

    num_img_per_prompt = 3

    print("image inpainting")
    inpainting_model = f"{model_dir}/stable-diffusion-inpainting"
    if not os.path.exists(inpainting_model):
        raise Exception(f"inpainting model does not exist: {inpainting_model}")
    inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(f"{model_dir}/stable-diffusion-inpainting")
    inpainting_pipe = inpainting_pipe.to("cuda")
    
    image_dir = '/home/azureuser/cloudfiles/code/projects/clir/hackathon51/'
    image_list = glob.glob(os.path.join(image_dir, 'image_data/noDamage/*.jpg'))
    masked_dir = 'masked'
    pipe_params = {'strength': 1.0,
                   'num_inference_steps': 300}
    prompts = ['car, crushed metal, after a car crash',
               'car, cracked glass after a car crash',
               'car, crushed door, broken glass, after a car crash, crumpled, dented']
    
    # For Danny
    image_list.sort(reverse=True)
    
    for image_file in image_list:
        masked_file = image_file.replace('image_data', masked_dir).replace('.jpg','.png')
        run_inpainting(image_file, masked_file, prompts=prompts, pipe_params=pipe_params)
        
    