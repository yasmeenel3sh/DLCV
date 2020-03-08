import numpy as np
import matplotlib.pyplot as plt
import random

# Yasmeen Khaled		37-6614		yasmeen.abdelmohsen@student.guc.edu.eg
# Michael George		37-3063		michael.khalil@student.guc.edu.eg
# Olfat Mostafa		37-19029	olfat.aaf@student.guc.edu.eg


# Generate an image with 4 quarters. Given a color range, the function splits the range 
# into 4 pixel values that used to be assigned to each quarter in the image.
def syn_quarter_image(height, width, num_color_range, sigma, noise_probability):

    syn_image = np.zeros([height,width], dtype=np.uint8)

    offset = num_color_range / 4

    # a,b,c,d will be used to represent 4 different pixel values
    a = offset / 2
    b = a + offset
    c = b + offset
    d = c + offset

    is_noise = False

    # Go over each pixel
    for y in range(0,height):
        for x in range(0,width):
            pixel_value = 0
            noise = 0
            is_noise = False

            # Generate noise if needed
            if sigma and noise_probability > 0:
                is_noise = True if random.random() <= noise_probability else False
                if is_noise:
                    noise = random.randint(-1*sigma, sigma)
            
            if x < width / 2:
                if y < height / 2:
                    pixel_value = a
                else:
                    pixel_value = b
            else:
                if y < height / 2:
                    pixel_value = c
                else:
                    pixel_value = d

            pixel_value += noise
            if pixel_value >= num_color_range:
                pixel_value = num_color_range - 1
            elif pixel_value < 0:
                pixel_value = 0

            syn_image[y][x] = pixel_value 

    return syn_image


# Generate an image splitted horizontally into two halves, each half with different shade of grey.
# Generate two vertical lines on top of that image with two other different shades 
def syn_image_lines(height, width, num_color_range, line1_start, line1_range, line2_start, line2_range, sigma, noise_probability):

    syn_image = np.zeros([height,width], dtype=np.uint8)

    offset = num_color_range / 4

    # a,b,c,d will be used to represent 4 different pixel values
    a = offset / 2
    b = a + offset
    c = b + offset
    d = c + offset

    is_noise = False

    # Go over each pixel
    for y in range(0,height):
        for x in range(0,width):
            pixel_value = 0
            noise = 0
            is_noise = False

            # Generate noise if needed
            if sigma and noise_probability > 0:
                is_noise = True if random.random() <= noise_probability else False
                if is_noise:
                    noise = random.randint(-1*sigma, sigma)
            
            if x >= line1_start and x <= line1_start + line1_range:
                pixel_value = c
            elif x >= line2_start and x <= line2_start + line2_range:
                pixel_value = d
            elif y < height / 2:
                pixel_value = a
            else:
                pixel_value = b

            pixel_value += noise
            if pixel_value >= num_color_range:
                pixel_value = num_color_range - 1
            elif pixel_value < 0:
                pixel_value = 0
    
            syn_image[y][x] = pixel_value

    return syn_image


