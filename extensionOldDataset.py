import os
import random
import tensorflow as tf 
from PIL import Image

def dataExtension (source_folder,changesOfImage,img_shape):        
    files = os.listdir(source_folder)   # List all files in the folder        
    files = [file for file in files if file.lower().endswith(('.png')) ] # Filter only png images
    for file in enumerate(files, start=1):
        old_name = file[1]                                   # file name, the first is a a number
        source_path = os.path.join(source_folder, old_name)  # file itself based on file name
        
        for i in range(0,changesOfImage): 
                new_name = "copy_" + str(i + 1) + "_" + old_name # copy name                 
                angle = random.randint(1, 359)                   # random angle, with degrees 1<=x<360  
                file= Image.open(source_path).rotate(angle)      # taking and rotating every image, randomly                                      
                file=file.resize(img_shape)                      # reshape to original size        
                destination_path = os.path.join(source_folder, new_name)
                file.save(destination_path)        

img_shape = (2560,1920)                                 # original size      
changesOfImage = 2                                      # how many changed copy we need 
source_path = 'Z:\VisualStudio\Python\TREND\old'        # where to save 
dataExtension(source_path,changesOfImage,img_shape)     # proceed


