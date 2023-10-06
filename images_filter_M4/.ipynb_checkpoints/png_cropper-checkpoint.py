from PIL import Image

def crop_images_to_border(start_index, end_index):
    for i in range(start_index, end_index + 1):
        input_image_path = f"D:\\Monash\\Y3S1\\ECE4078\\ECE4078-G3\\milestone4\\images_filter\\2_removedbg\\img_{i}.png"
        output_image_path = f"D:\\Monash\\Y3S1\\ECE4078\\ECE4078-G3\\milestone4\\images_filter\\2_removedbg\\img_{i}.png"

        # Open the image using Pillow
        try:
            img = Image.open(input_image_path)
        except FileNotFoundError:
            continue
        # Get the alpha (transparency) channel
        alpha = img.getchannel('A')

        # Get the non-zero (non-transparent) pixels
        non_transparent_pixels = alpha.point(lambda p: p > 0, '1')

        # Get the bounding box of the non-transparent pixels
        bbox = non_transparent_pixels.getbbox()

        # Crop the image to the bounding box
        cropped_image = img.crop(bbox)

        # Save the cropped image
        cropped_image.save(output_image_path)

if __name__ == "__main__":
    start_index = 1  # Replace with the starting index of your image files
    end_index = 14   # Replace with the ending index of your image files

    crop_images_to_border(start_index, end_index)
