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

def embed_text(dct_block, text, index):
    text_length = len(text)
    for i in range(8):
        for j in range(8):
            if i == 0 and j == 0 and dct_block[i, j] < 2:  # Skip DC component
                continue
            if index >= text_length:
                return dct_block, index  # Stop if all bits are embedded
            dct_block[i, j] = (dct_block[i, j] // 2) * 2 + int(text[index])  # Embed LSB
            index += 1
    return dct_block, index

def extract_text(dct_block):
    text = ''
    for i in range(8):
        for j in range(8):
            if i == 0 and j == 0 and dct_block[i, j] < 2:  # Skip DC component
                continue
            text += str(int(dct_block[i, j]) % 2)  # Extract LSB bit
    return text

def embed(input_image, output_image, message):
    """Embed a text message into an image using DCT-based steganography."""
    text = text_to_bits(message) + '00000000'  # Append end marker
    print(text)
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError("Image file not found.")

    height, width = img.shape
    img_dct = np.zeros((height, width), dtype=np.float32)

    index = 0  # Track bit position
    text_length = len(text)

    # Apply DCT and embed data into multiple blocks
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            img_dct[i:i+8, j:j+8] = dct_2d(img[i:i+8, j:j+8])


    for i in range(0, height, 8):
        for j in range(0, width, 8):
            dct_block = img_dct[i:i+8, j:j+8]
            img_dct[i:i+8, j:j+8], index = embed_text(img_dct[i:i+8, j:j+8], text, index)
            if index >= text_length:
                break  # Stop if all bits are embedded
        if index >= text_length:
            break

    # Convert back to spatial domain
    img_stego = np.zeros((height, width), dtype=np.uint8)
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            img_stego[i:i+8, j:j+8] = np.clip(idct_2d(img_dct[i:i+8, j:j+8]), 0, 255)

    cv2.imwrite(output_image, img_stego)
    print("Embedding complete! Saved to", output_image)

def extract(input_image):
    """Extract a hidden text message from a stego image using DCT-based steganography."""
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError("Image file not found.")

    height, width = img.shape
    text = ''

    # Apply DCT and extract data
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            dct_block = dct_2d(img[i:i+8, j:j+8])
            sub_text= extract_text(dct_block)
            text += sub_text
            if '00000000' in text:
                return text
    return ''

# Example usage
embed('a.jpg', 'b.jpg', "Hello, world!")
print(extract('b.jpg'))
