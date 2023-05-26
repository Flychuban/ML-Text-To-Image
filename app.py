import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

auth_token = os.getenv("AUTH_TOKEN")

app = tk.Tk()
app.geometry("532x622")
app.title("Stable Diffusion Demo")
ctk.set_appearance_mode('dark')

prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color='black', fg_color='white')
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(modelid, torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to('cpu')

def generate():
    with autocast('cpu'):
        image = pipe(prompt.get(), guidance_scale=8.5).images[0]
    
    image.save("generated_img.png")
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

trigger = ctk.CTkButton(app, text="Generate", height=40, width=120, font=("Arial", 20), fg_color='blue', text_color='white', command=generate)
trigger.place(x=206, y=60)

app.mainloop()