# CLIP+MLP Aesthetic Score Predictor

## セットアップ手順

```
pip install -r requirements.txt
```

## 実行方法

```
python simple_inference.py --img /path/to/file.png
```
標準出力にAesthetic Scoreが書き出される


## 以下misc

UPPER IS BETTER `convert -size 512x512 xc:#95a5a6 dummy.png` score is 4.398580074310303


Train, use and visualize an aesthetic score predictor ( how much people like on average an image ) based on a simple neural net that takes CLIP embeddings as inputs.


Link to the AVA training data ( already prepared) :
https://drive.google.com/drive/folders/186XiniJup5Rt9FXsHiAGWhgWz-nmCK_r?usp=sharing


Visualizations of all images from LAION 5B (english subset with 2.37B images) in 40 buckets with the model sac+logos+ava1-l14-linearMSE.pth:
http://captions.christoph-schuhmann.de/aesthetic_viz_laion_sac+logos+ava1-l14-linearMSE-en-2.37B.html


