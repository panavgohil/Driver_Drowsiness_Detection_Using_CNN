import cv2
import os

image_path = r"C:\Users\panav\Downloads\MRL\data\train\awake\s0001_01859_0_0_1_0_0_01.png"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

print("Shape:", img.shape)

print("Type:", type(img))

print("Top-left pixel:", img[0][0])