import streamlit as st
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import pretrainedmodels
from efficientnet_pytorch import EfficientNet

import json
with open("imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

available_models = [
    "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", 
    "efficientnet-b5", "efficientnet-b6", "efficientnet-b7", 
    "alexnet", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19_bn", "vgg19",
    "densenet121", "densenet169", "densenet201", "densenet161",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext101_32x4d", "resnext101_64x4d", 
    "squeezenet1_0", "squeezenet1_1", "nasnetamobile", "nasnetalarge", 
    "dpn68", "dpn68b", "dpn92", "dpn98", "dpn131", 
    "senet154", "se_resnet50", "se_resnet101", "se_resnet152", "se_resnext50_32x4d", "se_resnext101_32x4d",
    "inceptionv4", "inceptionresnetv2", "xception", "fbresnet152", "bninception",
    "cafferesnet101", "pnasnet5large", "polynet"
]
def load_moel(model_name):
    if "efficientnet" in model_name:
        model = EfficientNet.from_pretrained(model_name)    
    else:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000)
    return model



option = st.selectbox(
    'Select Model',
     available_models)

'You selected: ', option
st.text("Loading model....")
model = load_moel(option)
model.eval()
st.text("Complete model loading!")

# load data

img = Image.open("./img/kitten.jpg")
img_for_plot = np.array(img)
img = transforms.ToTensor()(img)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
img = normalize(img).unsqueeze(dim=0)
result = model(img).squeeze(dim=0)
predict_idx = result.argmax().item()
prob = torch.softmax(result, dim=0)
st.image(img_for_plot, use_column_width=True)
st.text(f"{idx2label[predict_idx]}, {prob[predict_idx]}")