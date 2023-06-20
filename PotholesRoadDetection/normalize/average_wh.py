import os
import cv2
import numpy as np
import math

image_directory = 'C:/Users/Grotti/Desktop/DATA/dataset/'

image_files = [file for file in os.listdir(image_directory) if file.endswith('.jpg') or file.endswith('.png')]

image_sizes = []

def get_image_size(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to read image: {image_path}, WH is None Type")
        return None
    height, width, _ = image.shape
    return height, width

def resize_image(image_path, new_height, new_width):
    image = cv2.imread(image_path)
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        print(f"Unable to resize image: {image_path} WH is None Type")
        return None
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    size = get_image_size(image_path)
    if size is not None:
        image_sizes.append(size)

avg_height = np.mean([size[0] for size in image_sizes])
avg_width = np.mean([size[1] for size in image_sizes])
print(f"AVG height = {avg_height}, AVG width = {avg_width}")

if avg_height or avg_width <= 640:
    print(f"AVG height or AVG width <= 640, HW multiply 2")
    avg_height = avg_height * 2
    avg_width = avg_width * 2

new_height = math.ceil((avg_height // 32) * 32) 
new_width = math.ceil((avg_width // 32) * 32) 
print(f"NEW height = {new_height}, NEW width = {new_width}")
print('RESIZEING IN PROCESS')

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    resized_image = resize_image(image_path, new_height, new_width)
    if resized_image is not None:
        cv2.imwrite(image_path, resized_image)
print(f'{len(image_files)} image files was resized to H:{new_height};W:{new_width}. ')
print('RESIZEING COMPLETE SUCCESSFULLY')




