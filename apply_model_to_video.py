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
INPUT_VIDEO_PATH = os.path.join(os.getcwd(), 'video.mp4')
OUTPUT_VIDEO_PATH = os.path.join(os.getcwd(), 'output.mp4')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.pth')
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
    # write the frame to the output video
    output.write(frame)
    # show image
    cv2.imshow('frame', frame)
    cv2.waitKey(1)



def main():
    # check if cuda is available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    # open the input video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    # get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # create the output video
    output = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    # get number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # loop through the frames
    for i in tqdm.tqdm(range(num_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        # apply mask rcnn to the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        detect_objects_in_frame(model, frame, device, output)
    # release the video
    cap.release()
    output.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
