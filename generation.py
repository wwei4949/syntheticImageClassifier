from diffusers import StableDiffusionPipeline
import csv
import json
import os
from pathlib import Path


def get_prompts_txt(txt_file):
    prompts = []
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()  # ignore empty lines
            if line:
                prompt = line.replace("speaker", "driver")
                # some prompts use the term 'speaker' to refer to the claimant (summarized from call logs)
                # it seems logical to adjust the context to refer to the individual as the 'driver'
                prompts.append(prompt)
    return prompts


def get_prompts_csv(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader, None)  # skip header
        prompts = [row[-1] for row in reader]
        prompts = [prompt for prompt in prompts if "no accident described" not in prompt.lower()]
    return prompts


if __name__ == "__main__":
    generation_prompts = []
    for prompt_file in ("prompts/damages.txt", "prompts/damages2.txt",
                        "prompts/summary_inpainting1.csv", "prompts/summary_generate1.csv"):
        if os.path.splitext(prompt_file)[-1].lower() == ".csv":
            prompts = get_prompts_csv(prompt_file)
        else:  # assume text file
            prompts = get_prompts_txt(prompt_file)
        print(f"extracted {len(prompts)} prompts from {prompt_file}")
        generation_prompts.extend([{"prompt": prompt, "source": prompt_file} for prompt in prompts])
    print(f"extracted a total of {len(generation_prompts)} prompts")

    # add an index to keep track of outputs
    for idx, prompt in enumerate(generation_prompts, start=0):
        prompt["identifier"] = idx

    with open("generation_prompts.json", "w") as f:
        json.dump(generation_prompts, f, indent=4)

    model_dir = "models"
    results_dir = "results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    generation_model = f"{model_dir}/stable-diffusion-v1-5"
    if not os.path.exists(generation_model):
        raise Exception(f"generation model does not exist: {generation_model}")
    generation_pipe = StableDiffusionPipeline.from_pretrained(generation_model)
    generation_pipe = generation_pipe.to("cuda")

    cnt = 1
    num_img_per_prompt = 2
    total_num_img_generated = num_img_per_prompt * len(generation_prompts)
    for i in range(0, num_img_per_prompt):
        for generation_prompt in generation_prompts:
            identifier = generation_prompt["identifier"]
            prompt = generation_prompt["prompt"]
            print(f"generating from prompt '{prompt}' [{i+1}/{num_img_per_prompt}] "
                  f"[{cnt}/{total_num_img_generated}]")
            image = generation_pipe(prompt).images[0]
            image.save(f"{results_dir}/{str(identifier).zfill(3)}_{str(i).zfill(2)}.png")
            cnt += 1
