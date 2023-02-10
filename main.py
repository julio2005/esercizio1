from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
# from rembg import remove
import os
import io
import warnings
# import cv2

from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import numpy as np
from torchvision.transforms import GaussianBlur

# Our host url should not be prepended with "https" nor should it have a trailing slash.
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'QUA IL TOKEN'

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None


@app.get("/")
def index():
    return {"message": "Hello World"}


@app.get("/list/{id}")
def list(id: int):
    return {"message": id}


@app.post("/create")
def create(item: Item):
    return {"item": item.name}


""" @app.post("/removebg")
def removebg():
    input_path = 'images/1.jpeg'
    output_path = 'images/output2.png'
    # cambiar esto con PIL
    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)
 """

@app.post("/stability")
def stability():
    # Set up our connection to the API.
    stability_api = client.StabilityInference(
        # API Key reference.
        key=os.environ['STABILITY_KEY'],
        verbose=True,  # Print debug messages.
        # Set the engine to use for generation. For SD 2.0 use "stable-diffusion-v2-0".
        engine="stable-diffusion-v1-5",
        # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
        # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
    )

    # Set up initial generation parameters, display image on generation, and safety warning for if the adult content classifier is tripped.

    answers = stability_api.generate(
        prompt="houston, we are a 'go' for launch!",
        # If a seed is provided, the resulting generated image will be deterministic.
        seed=34567,
        # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
        # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
        # Amount of inference steps performed on image generation. Defaults to 30.
        steps=30,
    )

    # Set up our warning to print to the console if the adult content classifier is tripped. If adult content classifier is not tripped, display generated image.
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                global img
                img = Image.open(io.BytesIO(artifact.binary))
                # Save our generated image with its seed number as the filename and the 1-start suffix so that we know this was our origin generation.
                img.save("images/" + str(artifact.seed) + "-1-start.png")


@app.post("/inpaint")
def inpaint():
    # Set up our connection to the API.
    stability_api = client.StabilityInference(
        # API Key reference.
        key=os.environ['STABILITY_KEY'],
        verbose=True,  # Print debug messages.
        # Set the engine to use for generation. For SD 2.0 use "stable-diffusion-v2-0".
        engine="stable-diffusion-v1-5",
        # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
        # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
    )
    answers3 = stability_api.generate(
        prompt="A photo of this open cream on top of a table.",
        init_image=Image.open("images/input.jpeg"),
        mask_image=Image.open("images/mask.png"),
        start_schedule=1,
        seed=1823948,  # If attempting to transform an image that was previously generated with our API,
        # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
        # Amount of inference steps performed on image generation. Defaults to 30.
        steps=30,
        # Influences how strongly your generation is guided to match your prompt.
        cfg_scale=8.0,
        # Setting this value higher increases the strength in which it tries to match your prompt.
        # Defaults to 7.0 if not specified.
        width=512,  # Generation width, defaults to 512 if not included.
        height=512,  # Generation height, defaults to 512 if not included.
        # Choose which sampler we want to denoise our generation with.
        sampler=generation.SAMPLER_K_DPMPP_2M
        # Defaults to k_lms if not specified. Clip Guidance only supports ancestral samplers.
        # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, display generated image.
    for resp in answers3:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                global img3
                img3 = Image.open(io.BytesIO(artifact.binary))
                # Save our completed image with its seed number as the filename, including the 5-completed suffix so that we know this is our final result.
                img3.save("images/" + str(artifact.seed) + "-5-completed.png")


@app.post("/addgrayscale")
def addgrayscale():
   # Load the PNG image
    img = cv2.imread("images/output.png", cv2.IMREAD_UNCHANGED)

    # Create a mask from the transparent pixels
    mask = (img[:, :, 3] == 0)

    # Replace the transparent pixels with black
    img[mask] = [0, 0, 0, 255]

    # Replace the non-transparent pixels with white
    img[np.logical_not(mask)] = [255, 255, 255, 255]

    # Save the processed image
    cv2.imwrite("images/processed_image.png", img)


@app.post("/addgrayscale")
def addgrayscale():
   # Load the PNG image
    img = cv2.imread("images/output.png", cv2.IMREAD_UNCHANGED)

    # Create a mask from the transparent pixels
    mask = (img[:, :, 3] == 0)

    # Replace the transparent pixels with black
    img[mask] = [0, 0, 0, 255]

    # Replace the non-transparent pixels with white
    img[np.logical_not(mask)] = [255, 255, 255, 255]

    # Save the processed image
    cv2.imwrite("images/processed_image.png", img)


@app.post("/addgrayscale2")
def addgrayscale2():
    # Load the PNG image
    img = cv2.imread("images/output2.png", cv2.IMREAD_UNCHANGED)

    # Create a mask from the transparent pixels
    mask = (img[:, :, 3] == 0)

    # Replace the transparent pixels with black
    img[mask] = [0, 0, 0, 255]

    # Replace the non-transparent pixels with white
    img[np.logical_not(mask)] = [255, 255, 255, 255]

    # Apply morphological operations to smooth the edges
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)

    # Save the processed image
    cv2.imwrite("images/processed_image2.png", img)


@app.post("/imgtoimg")
def imgtoimg():
    # Set up our connection to the API.
    stability_api = client.StabilityInference(
        # API Key reference.
        key=os.environ['STABILITY_KEY'],
        verbose=True,  # Print debug messages.
        # Set the engine to use for generation. For SD 2.0 use "stable-diffusion-v2-0".
        engine="stable-diffusion-v1-5",
        # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
        # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
    )
    # Set up our initial generation parameters.
    answers2 = stability_api.generate(
        prompt="A photo of this open cream on top of a table.",
        # Assign our previously generated img as our Initial Image for transformation.
        init_image=Image.open("images/input.jpeg"),
        # Set the strength of our prompt in relation to our initial image.
        start_schedule=0.6,
        # If attempting to transform an image that was previously generated with our API,
        seed=123467458,
        # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
        # Amount of inference steps performed on image generation. Defaults to 30.
        steps=30,
        # Influences how strongly your generation is guided to match your prompt.
        cfg_scale=8.0,
        # Setting this value higher increases the strength in which it tries to match your prompt.
        # Defaults to 7.0 if not specified.
        width=512,  # Generation width, defaults to 512 if not included.
        height=512,  # Generation height, defaults to 512 if not included.
        # Choose which sampler we want to denoise our generation with.
        sampler=generation.SAMPLER_K_DPMPP_2M
        # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
        # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, display generated image.
    for resp in answers2:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                global img2
                img2 = Image.open(io.BytesIO(artifact.binary))
                # Save our generated image with its seed number as the filename and the img2img suffix so that we know this is our transformed image.
                img2.save("images/" + str(artifact.seed) + "-img2img.png")