import torch
import torchvision
import requests
from io import BytesIO
import base64
from PIL import Image

# Load the pretrained model
model = torch.load('resnet18.pth')

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
model.eval()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_feature(image,output_type='tensor'):
    # Create a PyTorch tensor with the transformed image
    t_img = transforms(image)
    # Create a vector of zeros that will hold our feature vector
    # The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)

    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())                 # <-- flatten

    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # Run the model on our transformed image
    with torch.no_grad():                               # <-- no_grad context
        model(t_img.unsqueeze(0))                       # <-- unsqueeze
    # Detach our copy function from the layer
    h.remove()
    # Return the feature vector
    if output_type == 'tensor':
        return my_embedding
    elif output_type == 'numpy':
        return my_embedding.numpy()
    elif output_type == 'list':
        return my_embedding.numpy().tolist()


def pil_img_postprocessing(pil_img):
    if len(pil_img.mode) != 3:
        pil_img = pil_img.convert('RGB')
    return pil_img

def read_from_url(url):
    response = requests.get(url)
    return pil_img_postprocessing(Image.open(BytesIO(response.content)))


def read_from_b64(b64_str):
    pil_img = Image.open(BytesIO(base64.b64decode(b64_str)))
    return pil_img_postprocessing(pil_img)


def lambda_handler(event, context):
    # warmup 용 요청일 경우
    if event.get('warmup'):
        return {'statusCode':200,'message':'OK'}
    if event.get('url'):
        pil_image = read_from_url(event['url'])
    elif event.get('imageData'):  # b64
        pil_image = read_from_b64(event['imageData'])
    else:
        raise ValueError("No Valid url/imageData Parameter Offered!")

    feature = get_feature(pil_image,output_type='list')

    return {'statusCode': 200, 'message': "success", 'feature': feature}

