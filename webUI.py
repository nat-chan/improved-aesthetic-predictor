from PIL import Image
import torch
import clip
from PIL import Image
from simple_inference import MLP, normalized
import gradio as gr
import numpy as np

model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
s = torch.load("sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
model.load_state_dict(s)
model.to("cuda")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   

def single_run(pil_image: Image) -> float:
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model2.encode_image(image)
    im_emb_arr = normalized( image_features.cpu().detach().numpy() )
    input_tensor = torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
    prediction = model(input_tensor)
    return float(prediction[0])

def single_run_ui(np_image: np.ndarray) -> str:
    pil_image = Image.fromarray(np_image)
    aesthetic_score = single_run(pil_image)
    return str(aesthetic_score)


with gr.Blocks(title="抽象背景生成", css="footer {visibility: hidden}") as demo:
    with gr.Row():
        image_single = gr.Image(label="aesthetic scoreを計りたい１枚の入力画像")
        text_single = gr.Textbox(label="計算したaesthetic scoreの出力")
    button_single = gr.Button("1枚の画像に対してaesthetic scoreを計算する")
    button_single.click(
        fn=single_run_ui, 
        inputs=[image_single],
        outputs=[text_single],
    )

        


demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)
