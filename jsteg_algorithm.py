"""
------------------------- Jsteg Algorithm ------------------------
- DCT Transformation: Applies Discrete Cosine Transform (DCT) to divide the image into 8×8 blocks for frequency domain manipulation.
- Embedding Process: Modifies low-frequency DCT coefficients (excluding DC component) to store text bits while minimizing visual impact.
- Retrieves the embedded bits from selected DCT coefficients.
"""


import cv2
import numpy as np

def text_to_bits(text):
    return ''.join(f'{ord(c):08b}' for c in text)

def bits_to_text(bits):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(c, 2)) for c in chars if int(c, 2) != 0)

def dct_2d(block):
    return cv2.dct(np.float32(block))

def idct_2d(block):
    return cv2.idct(block)

def embed_text(img_dct, text):
    text_bits = [int(bit) for bit in text]
    index = 0
    height, width = img_dct.shape
    
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if index >= len(text_bits):
                return img_dct  # End if all data is embedded
            
            for x in range(1, 8):  # Skip DC (0,0)
                for y in range(1, 8):
                    if index < len(text_bits):
                        img_dct[i + x, j + y] = (img_dct[i + x, j + y] // 2) * 2 + text_bits[index]
                        index += 1
                    else:
                        return img_dct  
    return img_dct

def extract_text(img_dct, length):
    extracted_bits = []
    height, width = img_dct.shape
    
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if len(extracted_bits) >= length:
                break
            
            for x in range(1, 8):
                for y in range(1, 8):
                    if len(extracted_bits) < length :
                        extracted_bits.append(int(img_dct[i + x, j + y]) % 2)
                    else:
                        break
    return ''.join(map(str, extracted_bits))

def embed_and_extract(input, text):

    text = text_to_bits(text)

    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    # DCT transform for each 8x8 block
    img_dct = np.zeros((height, width), dtype=np.float32)
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            img_dct[i:i+8, j:j+8] = dct_2d(img[i:i+8, j:j+8])

    
    img_dct = embed_text(img_dct, text)

    extracted_text = extract_text(img_dct, len(text))
    print(f"Extracted text: {bits_to_text(extracted_text)}")

# Run 
# embed_and_extract(input_path, text)
