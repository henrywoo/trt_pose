#!/usr/bin/python

from torch2trt import TRTModule
import torch
import json
import trt_pose.coco
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from jetcam.usb_camera import USBCamera
# from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg


with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

WIDTH = 224
HEIGHT = 224

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

width, height = ( 
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )   
print(f"Camera dimensions: {width, height}")
print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")

import time
count= 0
start = time.monotonic()
while 1:
    ret, image = cap.read()
    if ret and 0:
        image = cv2.resize(image,(224,224))

    if 0:
        start = time.monotonic()
        data = preprocess(image)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        draw_objects(image, counts, objects, peaks)

    if False:
        cv2.imshow('frame', cv2.resize(image,(640,480)))
        if cv2.waitKey(1) == ord('q'):
            break
    count+=1
    if count%30==0:
        end = time.monotonic()
        print("FPS:", 30/(end-start))
        start = end

cap.release()
cv2.destroyAllWindows()

