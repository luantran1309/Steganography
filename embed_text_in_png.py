# Use the LSB bit in the PNG file to embed text, as it will not significantly change the image's value.

from PIL import Image
import numpy as np

def text_to_bits(text):
    return ''.join(f'{ord(c):08b}' for c in text)

def bits_to_text(bits):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(c, 2)) for c in chars)

def embed_text_in_png(image_path, output_path, text):
    img = Image.open(image_path)
    img_data = np.array(img)
    
    bits = text_to_bits(text) + '00000000'  # Add 8 bits 0 to stop
    flat_data = img_data.flatten()
    
    if len(bits) > len(flat_data):
        raise ValueError("Image is too small to embed text!")
    
    for i in range(len(bits)):
        flat_data[i] = (flat_data[i] & 0b11111110) | int(bits[i])
    
    img_data = flat_data.reshape(img_data.shape)
    encoded_img = Image.fromarray(img_data)
    encoded_img.save(output_path)
    print("Embed text success!")

def extract_text_from_png(image_path):
    img = Image.open(image_path)
    img_data = np.array(img)
    flat_data = img_data.flatten()
    
    bits = ""

    for byte in flat_data:
        bit = str(byte & 1)  # Take LSB
        bits += bit
        if len(bits) % 8 == 0 and bits[-8:] == '00000000':
            break


    text = bits_to_text(bits[:-8]) 
    return text


image_file = "input.png"
output_file = "output.png"
text_to_hide = "hello world"

embed_text_in_png(image_file, output_file, text_to_hide)
extracted_text = extract_text_from_png(output_file)
print("Extract text from image:", extracted_text)
