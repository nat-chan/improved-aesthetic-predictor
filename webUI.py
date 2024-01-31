from PIL import Image
import torch
import clip
from PIL import Image
from simple_inference import MLP, normalized
import gradio as gr
import numpy as np
import os
import zipfile
from zipfile import ZipFile
import tempfile

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

def extract_image_files_from_zip(zip_path: str, tmp_dir: str) -> list[str]:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    
    image_files = list()
    for root, dirs, files in os.walk(tmp_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    return image_files

def multi_run_ui(file_obj: tempfile._TemporaryFileWrapper) -> str:
    outputs = list()
    with ZipFile(file_obj.name) as zfile:
        for zinfo in zfile.infolist():
            if not zinfo.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            with zfile.open(zinfo.filename) as f:
                pil_image = Image.open(f)
                score = single_run(pil_image)
                line = f"{score}, {zinfo.filename}"
                outputs.append(line)
                print(line)
    return "\n".join(outputs)


with gr.Blocks(title="抽象背景生成", css="footer {visibility: hidden}") as demo:
    gr.Markdown("## 単一の画像に対する処理")
    with gr.Row():
        image_single = gr.Image(label="aesthetic scoreを計算したい１枚の入力画像")
        text_single = gr.Textbox(label="計算したaesthetic scoreの出力")
    button_single = gr.Button("1枚の画像に対してaesthetic scoreを計算する")
    button_single.click(
        fn=single_run_ui, 
        inputs=[image_single],
        outputs=[text_single],
    )
    gr.Markdown("## 複数の画像zipに対する処理")
    with gr.Row():
        zip_multi = gr.File(label="aesthetic scoreを計りたい複数枚の入力画像のzip")
        text_multi = gr.Textbox(label="計算したaesthetic scoreとファイルパスの出力列")
    button_multi = gr.Button("複数枚の画像zipに対してaesthetic scoreを計算する")
    button_multi.click(
        fn=multi_run_ui, 
        inputs=[zip_multi],
        outputs=[text_multi],
    )

        


demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)
