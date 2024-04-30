from llama_cpp import Llama
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import time

class Artist:
    sd_model_id = "kandinsky-community/kandinsky-2-2-decoder"
    llm_model_id = "TheBloke/MythoMist-7B-GGUF"
    llm_file_name = "mythomist-7b.Q5_K_M.gguf"

    llm = None
    pipe = None

    seed_prompt = 'An example of a prompt for Stable Diffusion is "'
    
    def __init__(self, no_sd=False, no_llm=False) -> None:
        if not no_llm:
            print("Initializing llm...") 
            self.llm = Llama.from_pretrained(
                repo_id=self.llm_model_id,
                filename=self.llm_file_name,
                # seed=12345, # Uncomment to set a specific seed
                n_ctx=512,
                n_batch=512,
                verbose=False,
            )

        if not no_sd:
            print("Initializing sd...") 
            self.pipe = AutoPipelineForText2Image.from_pretrained(self.sd_model_id, low_cpu_mem_usage=True)
            self.pipe = self.pipe.to("cpu")


    def generate_prompt(self):
        resp = self.llm(
        prompt=self.seed_prompt, # Prompt
        max_tokens=77, # Generate up to 32 tokens, set to None to generate up to the end of the context window
        stop=['"'], # Stop generating just before the model would generate a new question
        echo=False # Echo the prompt back in the output
        )
        return resp["choices"][0]["text"]


    def generate_image(self, prompt):
        negative_prompt = "low quality, bad quality, deformed"

        image = self.pipe(prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    prior_guidance_scale =1,
                    num_inference_steps=50, 
                    width=400, height=400, 
                    #callback_on_step_end=decode_tensors, 
                    #callback_on_step_end_tensor_inputs=["latents"]
                    ).images[0]

        return image
    
    def save_output(self, prompt, name, image):
        image_name = "output/{name}".format(name=name)
        print("Saving image as '{image_name}'".format(image_name=image_name))
        image.save(image_name+".png")
        f = open(image_name+"-prompt.txt", "a")
        f.write(prompt)
        f.close()

    def make_art(self, timestamp=None):
        if timestamp == None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")

        print("Generating prompt....\n")
        prompt = self.generate_prompt()
        print(prompt)
        print("Generating image...")
        image = self.generate_image(prompt)
        print("Saving output...")
        self.save_output(prompt,timestamp,image)
        print("Done!")

    def decode_tensors(self,pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        
        image = self.latents_to_rgb(latents)
        image.save(f"inprogress/{step}.png")

        return callback_kwargs

    def latents_to_rgb(self,latents):
        weights = (
            (60, -60, 25, -70),
            (60,  -5, 15, -50),
            (60,  10, -5, -35)
        )

        weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
        biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
        rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
        image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
        image_array = image_array.transpose(1, 2, 0)

        return Image.fromarray(image_array)


