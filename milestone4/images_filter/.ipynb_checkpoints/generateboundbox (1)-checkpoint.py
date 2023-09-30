from PIL import Image, ImageDraw
from random import random, randint
import os 
import numpy as np


num_of_dataset = 5000 #Number of images you want to generate
list_of_fruit = [['capsicum', 81], ['orange', 80], ['redapple', 80], ['greenapple', 80], ['mango', 160]]  #Number of cropped images of each type of fruit
num_of_background = 76  #Number of background images

# Iterate for the desired number of datasets
for n in range(num_of_dataset):
    num_of_fruit = randint(1, 3)
    # Create a set to keep track of placed fruits in this image
    placed_fruits = set()
    
    # Create a text file to store the coordinates of fruits with the same name as the corresponding dataset image
    # f = open(f"C:\\Users\\sshya\\Desktop\\Uni_2023\\ECE4078\\labels\\img_{n}.txt", "w+") 
    f = open(f"D:/Monash/Y3S1/ECE4078/ECE4078-G3/milestone4/images_filter/5_NN_Ready/labels/coord_{n}.txt", "w+") 
    
    # Randomly select a background
    bg_index = randint(1, 60)
    # bg = Image.open(f"C:\\Users\\sshya\\Desktop\\Uni_2023\\ECE4078\\arena_imgs\\calib_{bg_index}.jpg")  
    # bg = Image.open(f"milestone4\\images_filter\\4_arena_pics\\arena_{bg_index}.png")  
    input_background = "./4_arena_pics/"
    input_bg_list = os.listdir(input_background)
    img_path_bg = os.path.join(input_background, input_bg_list[bg_index ])
    bg = Image.open(img_path_bg)

    for m in range(num_of_fruit):
    # Choose a fruit type that hasn't been placed in this image yet
        while True:
            fruit_type = randint(0, 4)
            if fruit_type not in placed_fruits:
                placed_fruits.add(fruit_type)
                break
    
        i = randint(1, list_of_fruit[fruit_type][1]) # Choose a random picture of the selected fruit

        # fruit_path = os.path.join(os.getcwd(), f'Desktop\\Uni_2023\\ECE4078\\croppedfruits', str(list_of_fruit[fruit_type][0]))
        fruit_path = os.path.join(os.getcwd(), f'./2_removedbg', str(list_of_fruit[fruit_type][0]))
        img_path = os.path.join(fruit_path, f'img_{i}.png')  # Corrected line

        fruit = Image.open(img_path).convert("RGBA")

        imgx = bg  # Store the background in a separate variable

        bg_width, bg_height = imgx.size
        width, height = fruit.size
        aspect = width / height
        x = np.ceil(bg_width / num_of_fruit)

        upper_thres = np.floor(aspect * bg_height)
        if upper_thres > x:
            upper_thres = x
        random_width = randint(100, upper_thres)
        random_height = np.floor(random_width / aspect)
        new_size = (int(random_width), int(random_height))
        img4 = fruit.resize(new_size)

        random_location = (randint(x * m, x * m + 1), randint(0, bg_height - random_height))
        c1 = (random_location[0] + random_width / 2) / bg_width
        c2 = (random_location[1] + random_height / 2) / bg_height
        s1 = random_width / bg_width
        s2 = random_height / bg_height

        imgx.paste(img4, random_location, mask=img4)
        f.write(f"{str(fruit_type)} {c1} {c2} {s1} {s2}\n")
        # imgx.save(f"C:\\Users\\sshya\\Desktop\\Uni_2023\\ECE4078\\output\\img_{n}.jpg") 
        imgx.save(f"D:\Monash\Y3S1\ECE4078\ECE4078-G3\milestone4\images_filter\5_NN_Ready\images\img{n}.png") 


    f.close()
