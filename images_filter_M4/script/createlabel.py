import os
from PIL import Image

 
# def convertImage():
#     img = Image.open("./Apple_NoBg.png")
#     img = img.convert("RGBA")
 
#     datas = img.getdata()
#     images_to_collect =
    
#     newImage = []

#     for i in range(images_to_collect):
#         filename = "calib_{}.png".format(i)
#         savefilename = "calib_{}_label.png".format(i)
#         for item in datas:
#             if item[0] == 255 and item[1] == 255 and item[2] == 255:
#                 newImage.append(0)
#             else:
#                 newImage.append(1)
 
#         img.putdata(newImage)
#         img.save(savefilename, "PNG")
#         #cv2.imwrite(savefilename, newImage)
 
# convertImage()

def convertImages():
    input_directory = "./images/"
    output_directory = "./labels/"
    # input_directory = "./milestone3\script\images"
    # output_directory = "./milestone3\script\labels"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):
            img_path = os.path.join(input_directory, filename)
            img = Image.open(img_path)
            img = img.convert("RGBA")

            datas = img.getdata()
            width, height = img.size

            newImage = Image.new("L", (width, height))

            for i, item in enumerate(datas):
                if item[3] == 0:  # Check if alpha (transparency) value is 0
                    newImage.putpixel((i % width, i // width), 0)
                else:
                    if (i <= 17):
                        labelValue = 1
                    elif (i > 17 & i <= 32):
                        labelValue = 5
                    elif (i > 32 & i <= 64):
                        labelValue = 2
                    elif (i > 65 & i <= 115):
                        labelValue = 4
                    elif (i > 115):
                        labelValue = 3
                    
                    
                    newImage.putpixel((i % width, i // width), 255)

            base_filename, extension = os.path.splitext(filename)
            save_filename = os.path.join(output_directory, base_filename + "_label" + extension)
            newImage.save(save_filename, "PNG")
            print("Saved", save_filename)

    print("All images processed")

convertImages()