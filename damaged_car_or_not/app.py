import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('model.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Damaged Vehicle Classifier"
description = "A damaged vehicle classifier to identify vehicles especially cars that have been through accidents. Trained on some random images I downloaded from DuckDuckGo"
interpretation='default'
enable_queue=True

demo = gr.Interface(fn=predict,inputs="image",outputs="label",title=title,description=description,interpretation=interpretation,enable_queue=enable_queue).launch()
demo.launch()