os.system("pip install --upgrade pip")
os.system("pip install umap-learn bechdelai==0.0.1a2")

import gradio as gr
import os

from bechdelai.image.face_detection import FacesDetector
from bechdelai.image.gender_detection import GenderDetector
from bechdelai.image.vilt import ViLT
from bechdelai.image.img import Img

fd = FacesDetector()
gd = GenderDetector()

examples_path = "static/examples"
examples_images = [os.path.join(examples_path,x) for x in os.listdir(examples_path)]

def analyze_genders(img):
    rois,faces = fd.detect(img,method = "retinaface",padding = 20)
    probas = gd.predict(faces)
    results,metrics = gd.analyze_probas(probas)
    img_with_faces = fd.show_faces_on_image(img,rois,width = 3,genders = probas["gender"])

    new_metrics = {"Space occupied by women":metrics["women_area"],"Space occupied by men":metrics["men_area"]}

    return new_metrics, img_with_faces


demo = gr.Interface(
    fn = analyze_genders, 
    inputs = gr.Image(), 
    outputs = [
        gr.outputs.Label(num_top_classes=3),
        "image",
    ],
    examples=examples_images,
    title="BechdelAI Tool - Image Analysis",
    description="BechdelAI automates the Bechdel test and study of women under-representation in cinema and media with AI",
    article="For more information visit https://github.com/dataforgoodfr/bechdelai"
)
demo.launch()