
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def load_input(batch=1):
    # Download an example image from the pytorch website
    #import urllib
    #url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    #try: urllib.URLopener().retrieve(url, filename)
    #except: urllib.request.urlretrieve(url, filename)
    filename = "dog.jpg"
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
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        print("batch", i)
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())
        print("")
