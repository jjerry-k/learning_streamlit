import streamlit as st

import json
from io import BytesIO

import numpy as np
from PIL import Image

import torch
from torchvision import models
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

# =====================================
# =====================================
#           GLOBAL VARIABLE
# =====================================
# =====================================

# About Classification
classification_models = {
    "alexnet": models.alexnet, "vgg11": models.vgg11, "vgg11_bn": models.vgg11_bn, "vgg13": models.vgg13, "vgg13_bn": models.vgg13_bn, 
    "vgg16": models.vgg16, "vgg16_bn": models.vgg16_bn, "vgg19_bn": models.vgg19_bn, "vgg19": models.vgg19,
    "densenet121": models.densenet121, "densenet169": models.densenet169, "densenet201": models.densenet201, "densenet161": models.densenet161,
    "resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50, "resnet101": models.resnet101, "resnet152": models.resnet152, 
    "resnext101_32x8d": models.resnext101_32x8d, "resnext50_32x4d": models.resnext50_32x4d, "wide_resnet50_2": models.wide_resnet50_2, "wide_resnet101_2": models.wide_resnet101_2,
    "squeezenet1_0": models.squeezenet1_0, "squeezenet1_1": models.squeezenet1_1,
    "shufflenet_v2_x0_5": models.shufflenet_v2_x0_5, "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0, "shufflenet_v2_x1_5": models.shufflenet_v2_x1_5, "shufflenet_v2_x2_0": models.shufflenet_v2_x2_0,
    'mnasnet0_5': models.mnasnet0_5, 'mnasnet0_75': models.mnasnet0_75, 'mnasnet1_0': models.mnasnet1_0, 'mnasnet1_3': models.mnasnet1_3,
    'googlenet': models.googlenet, 'inception_v3': models.inception_v3, 'mobilenet_v2': models.mobilenet_v2, 'mobilenet_v3_large': models.mobilenet_v3_large, 'mobilenet_v3_small': models.mobilenet_v3_small
}
with open("imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# About Detection
detection_models = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "fasterrcnn_mobilenet_v3_large_fpn": models.detection.fasterrcnn_mobilenet_v3_large_fpn, 
    "fasterrcnn_resnet50_fpn": models.detection.fasterrcnn_resnet50_fpn, 
    # "maskrcnn_resnet50_fpn": models.detection.maskrcnn_resnet50_fpn, 
    "retinanet_resnet50_fpn": models.detection.retinanet_resnet50_fpn, 
    "ssdlite320_mobilenet_v3_large": models.detection.ssdlite320_mobilenet_v3_large
}
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
CLEAN_INSTANCE_CATEGORY_NAMES = sorted([
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',  'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
])

# About Segmentation
segmentation_models = {
    "deeplabv3_mobilenet_v3_large": models.segmentation.deeplabv3_mobilenet_v3_large,  
    "deeplabv3_resnet101": models.segmentation.deeplabv3_resnet101, 
    "deeplabv3_resnet50": models.segmentation.deeplabv3_resnet50, 
    "fcn_resnet101": models.segmentation.fcn_resnet101, 
    "fcn_resnet50": models.segmentation.fcn_resnet50, 
    "lraspp_mobilenet_v3_large": models.segmentation.lraspp_mobilenet_v3_large
}
sem_classes = [
            '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

# =====================================
# =====================================
#               FUNCTION
# =====================================
# =====================================

def mode_selector(mode):
    if mode.lower()=="classification":
        option = st.sidebar.selectbox(
                "Select Model", 
                classification_models)

    elif mode.lower()=="detection":
        option = st.sidebar.selectbox(
                "Select Model",
                detection_models)

    else:
        option = st.sidebar.selectbox(
                "Select Model",
                segmentation_models)

    return option

def set_controller(mode):
    if mode.lower()=="classification":
        option = None

    elif mode.lower()=="detection":
        class_select = st.sidebar.selectbox("Select Class", CLEAN_INSTANCE_CATEGORY_NAMES)
        conf = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
        option = [class_select, conf]

    else:
        class_select = st.sidebar.selectbox("Select Class", sem_classes)
        conf = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
        option = [class_select, conf]
        
    return option

def load_model(mode, model_name):
    
    if mode.lower()=="classification":
        model = classification_models[model_name](pretrained=True, progress=False)
    elif mode.lower()=="detection":
        model = detection_models[model_name](pretrained=True, progress=False)
    else:
        model = segmentation_models[model_name](pretrained=True, progress=False)
    return model
    
def load_image(bytes_data, mode):
    img = Image.open(BytesIO(bytes_data)).convert("RGB")
    plot_img = np.array(img)
    img = torch.Tensor(plot_img).type(torch.uint8).permute(2,0,1)
    batch_img = img.unsqueeze(dim=0)/255.
    if mode.lower() != "detection":
        batch_img = F.normalize(batch_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return img, batch_img

def inference(image, prediction, mode, option):
    result = None
    if mode.lower()=="classification":
        # Top 1~3
        prediction = prediction.squeeze(dim=0)
        predict_idx = prediction.argmax().item()
    
        prob = torch.softmax(prediction, dim=0)
        st.image(image.numpy().transpose(1, 2 ,0))
        st.text(f"{idx2label[predict_idx]}, {prob[predict_idx]}")
        result = None

    elif mode.lower()=="detection":
        prediction = prediction[0]
        valid_box_list = (prediction["scores"] > option[1]) * (prediction["labels"] == COCO_INSTANCE_CATEGORY_NAMES.index(option[0]))
        num_box = valid_box_list.sum()
        labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in prediction['labels'][valid_box_list]]
        img_with_boxes = draw_bounding_boxes(image, boxes=prediction['boxes'][valid_box_list], labels=labels, colors = [(126, 200, 128)] * num_box, fill=True, width=4)
        img_with_boxes = img_with_boxes.numpy().transpose(1,2,0)
        result = img_with_boxes
        st.image(result, use_column_width=True)

    else:
        normalized_masks = torch.nn.functional.softmax(prediction['out'], dim=1)
        num_classes = normalized_masks.shape[1]
        masks = normalized_masks[0]
        class_dim = 0
        all_classes_masks = masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None]
        img_with_all_masks = draw_segmentation_masks(image, masks=all_classes_masks[sem_class_to_idx[option[0]]], alpha=.6)
        img_with_all_masks = img_with_all_masks.numpy().transpose(1,2,0)
        result = img_with_all_masks
        st.image(result, use_column_width=True)
    return result

def main():
    # global model
    st.sidebar.title(f"Pretrained Model Test")
    mode = st.sidebar.selectbox("Choose the app mode",
            ["Classification", "Detection", "Segmentation"])

    model_name = mode_selector(mode)

    model = load_model(mode, model_name)
    model.eval()

    uploaded_file = st.sidebar.file_uploader("Choose a Image")
    option = set_controller(mode)
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        img, batch_img = load_image(bytes_data, mode)
        pred = model(batch_img)
        inference(img, pred, mode, option)
        
if __name__ == "__main__":
    main()