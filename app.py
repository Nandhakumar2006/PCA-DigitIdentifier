import pickle
import gradio as gr
import numpy as np
from PIL import Image


with open("digit_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

def predict_manual(*features):
    features = np.array(features, dtype=float).reshape(1, -1)
    prediction = pipeline.predict(features)
    return int(prediction[0])

def predict_image(img):

    display_img = img.resize((200, 200))
    

    img_small = img.convert("L").resize((8, 8), resample=Image.Resampling.LANCZOS)
    img_array = np.array(img_small, dtype=float)

    img_array = (16 - (img_array / 255.0 * 16)).flatten().reshape(1, -1)
    
    prediction = pipeline.predict(img_array)
    return display_img, int(prediction[0])

manual_inputs = [gr.Number(label=f"Pixel {i}", value=0) for i in range(64)]

with gr.Blocks(title="Digit Recognition App") as demo:
    gr.Markdown("## Digit Recognition using PCA + KNN Pipeline")
    

    with gr.Tab("Manual Input"):
        gr.Markdown("Enter the 64 pixel values manually (0–16 scale like sklearn digits):")
        for inp in manual_inputs:
            inp.render()
        manual_output = gr.Label(label="Predicted Digit")
        manual_button = gr.Button("Predict Digit")
        manual_button.click(fn=predict_manual, inputs=manual_inputs, outputs=manual_output)
 
    with gr.Tab("Upload Image"):
        gr.Markdown("Upload an image of a digit (grayscale or color):")
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Digit Image")
            with gr.Column():
                image_output = gr.Image(label="Resized Image for Display")
                digit_output = gr.Label(label="Predicted Digit")
        
        image_button = gr.Button("Predict Digit")
        image_button.click(fn=predict_image, inputs=image_input, outputs=[image_output, digit_output])

demo.launch()
