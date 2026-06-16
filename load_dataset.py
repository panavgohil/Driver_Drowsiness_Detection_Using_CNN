import os
import cv2
images=[]
labels=[]
open_path = "dataset/open"
closed_path = "dataset/closed"
for filename in os.listdir(open_path):
    filepath=os.path.join(open_path, filename)
    img=cv2.imread(filepath)
    images.append(img)
    labels.append(0)
for filename in os.listdir(closed_path):
    filepath = os.path.join(closed_path, filename)
    img = cv2.imread(filepath)
    images.append(img)
    labels.append(1)
print("Total images= ", len(images))
print("Total labels= ", len(labels))