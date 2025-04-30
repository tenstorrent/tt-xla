
import torch
from torchvision import transforms
from PIL import Image
import os

def load_input(batch=1):
    # Download an example image from the pytorch website
    #import urllib
    #url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    #try: urllib.URLopener().retrieve(url, filename)
    #except: urllib.request.urlretrieve(url, filename)
    filename = os.path.dirname(os.path.abspath(__file__)) + "/dog.jpg"
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    input = torch.cat(batch*[input])
    return input

def print_output(output):
    for i in range(output.shape[0]):
        probabilities = torch.nn.functional.softmax(output[i], dim=0)
        #print(probabilities)

        # Read the categories
        with open(os.path.dirname(os.path.abspath(__file__)) + "/imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        print("batch", i)
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())
        print("")

def get_models():
    # availalbe models in torch.hub
    models = [
        "alexnet",
        "mobilenet_v2",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2"
    ]
    #"resnext101_64x4d",
    #"regnet_y_400mf",
    #"regnet_y_800mf",
    #"regnet_y_1_6gf",
    #"regnet_y_3_2gf",
    #"regnet_y_8gf",
    #"regnet_y_16gf",
    #"regnet_y_32gf",
    #"regnet_x_400mf",
    #"regnet_x_800mf",
    #"regnet_x_1_6gf",
    #"regnet_x_3_2gf",
    #"regnet_x_8gf",
    #"regnet_x_16gf",
    #"regnet_x_32gf",
    return models