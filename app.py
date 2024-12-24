import gradio as gr
import time
import cv2
from inpaint import Inpainting, get_args
import numpy as np
from PIL import Image
args = get_args()
print(args)
model_inpaint = Inpainting(args)
def predict(im):
    if im["background"].shape == 4:
        input_img = im["background"][:,:,:3]
    else:
        input_img = im["background"]
    input_img = Image.fromarray(input_img).convert("RGB")
    
    if len(im["layers"]) > 0:
        mask = im["layers"][0][:,:,-1]
        mask = 255 - mask
        mask[mask < 255] = 0
        input_mask = Image.fromarray(mask).convert("L")

    output_img = model_inpaint.inpaint(input_img, input_mask)
    return output_img

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.ImageMask(
            type="numpy",
        )
        output_img = gr.Image(type="numpy")

    predict_btn = gr.Button("Inpaint")
    predict_btn.click(predict, [input_img], [output_img])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)