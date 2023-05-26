import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

auth_token = os.getenv("AUTH_TOKEN")