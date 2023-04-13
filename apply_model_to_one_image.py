import cv2
import torch
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import tqdm
import os


# constants
INPUT_IMAGE_PATH = os.path.join(os.getcwd(), 'racoon dataset', 'test', 'images-JPG',
                                'from_internet_2.jpeg')
OUTPUT_IMAGE_PATH = os.path.join(os.getcwd(), 'from_internet_2_output.jpg')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'maskrcnn_resnet50_fpn_racoon_witout_backbone.pth')
classes = ['background', 'raccoon']

def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def detect_objects_in_frame(model, frame, device, output):
    # apply mask rcnn to the frame
    transform = get_transform()
    frame_tensor = transform(frame)
    frame_tensor = frame_tensor.to(device)
    with torch.no_grad():
        predictions = model([frame_tensor])
    # get the predictions
    boxes = predictions[0]['boxes'].cpu().detach()
    labels = predictions[0]['labels'].cpu().detach()
    scores = predictions[0]['scores'].cpu().detach()
    # add threshold
    threshold = 0.6
    boxes = boxes[scores > threshold]
    labels = labels[scores > threshold]
    # now labels is a tensor of size (n,) where n is the number of objects detected
    # we need to convert it to a list of strings
    labels = [classes[label] for label in labels]
    # this is a new tensor that is uint8 for the draw function and rgb
    frame_tensor = T.PILToTensor()(frame)
    # draw the bounding boxes
    frame_tensor = draw_bounding_boxes(frame_tensor, boxes, labels)
    # convert the tensor back to an image
    frame = T.ToPILImage()(frame_tensor)
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    # write the image to the output image
    cv2.imwrite(OUTPUT_IMAGE_PATH, frame)



def main():
    # check if cuda is available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    image = cv2.imread(INPUT_IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    # get the video writer
    output = image.copy()
    # detect objects in the frame
    detect_objects_in_frame(model, image, device, output)

if __name__ == '__main__':
    main()
