import cv2
import numpy as np
import os


def preproc_img(file_name,output_dir):
    image_name = file_name.split("/")[-1][:-4] #remove .png from the name

    # Read image
    img = cv2.imread(file_name,0)
    if img.shape != [28,28]:
        img2 = cv2.resize(img,(28,28))
        
    img = img2.reshape(28,28,-1);

    #revert the image,and normalize it to 0-1 range
    img = 1.0 - img/255.0

    save_file_name = os.path.join(output_dir, image_name + ".txt")
    # Save image in row major format (i.e. channel x height x width)
    img = img.reshape(1,28,28)

    # Each element is saved in a new line in row major with 6 decimal places (doing this so that its easy to read in C++ code)
    np.savetxt(save_file_name, img[0], fmt='%.6f', delimiter='\n')


input_dir = "./img"

output_dir = "./pre-proc-img"

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Add all files into a list
files = os.listdir(input_dir)

# Add full path to each file
files = [os.path.join(input_dir, f) for f in files]

# Loop through all files
for file in files:
    preproc_img(file,output_dir)

