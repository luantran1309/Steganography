'''
Script implements LSB (Least Significant Bit) steganography to embed and extract text in PNG images using two methods:
    - Ordered Mode: Hides text sequentially in pixel values.
    - Random Mode: Uses a key (a, p) to shuffle pixel positions before embedding.
'''

import argparse
from PIL import Image
import numpy as np

def text_to_bits(text):
    """Convert text to binary string."""
    return ''.join(f'{ord(c):08b}' for c in text)

def bits_to_text(bits):
    """Convert binary string to text."""
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(c, 2)) for c in chars)

def shuffle_indices(size, a, p):
    """Generate shuffled indices using yi = (a * i) % p formula."""
    return [(a * i) % p for i in range(size)]

def embed_text_in_png(image_path, output_path, text, mode="ordered", a=13):
    """Embed text into an image using ordered or randomized LSB steganography."""
    img = Image.open(image_path)
    img_data = np.array(img)
    
    p = img_data.size  # Total number of pixels

    bits = text_to_bits(text) + '00000000'  # Add 8-bit stop sequence
    flat_data = img_data.flatten()

    if len(bits) > len(flat_data):
        raise ValueError("Image is too small to embed text!")

    # Choose embedding method
    if mode == "ordered":
        indices = list(range(len(bits)))  # Sequential embedding
    elif mode == "random":
        indices = shuffle_indices(len(bits), a, p)  # Randomized embedding
    else:
        raise ValueError("Invalid mode! Choose 'ordered' or 'random'.")

    for i, bit in enumerate(bits):
        index = indices[i % len(indices)]  # Loop if indices < bits
        flat_data[index] = (flat_data[index] & 0b11111110) | int(bit)

    img_data = flat_data.reshape(img_data.shape)
    encoded_img = Image.fromarray(img_data)
    encoded_img.save(output_path)
    print(f"✅ Text embedded successfully into {output_path}")

def extract_text_from_png(image_path, mode="ordered", a=13):
    """Extract text from an image using LSB method until encountering '00000000'."""
    img = Image.open(image_path)
    img_data = np.array(img)
    flat_data = img_data.flatten()

    p = img_data.size  # Total number of pixels

    # Choose extraction method
    if mode == "ordered":
        indices = list(range(len(flat_data)))  # Sequential extraction
    elif mode == "random":
        indices = shuffle_indices(len(flat_data), a, p)  # Randomized extraction
    else:
        raise ValueError("Invalid mode! Choose 'ordered' or 'random'.")

    bits = ""
    for index in indices:
        bits += str(flat_data[index] & 1)  # Extract LSB
        if len(bits) % 8 == 0 and bits[-8:] == '00000000':  # Stop when '00000000' is found
            break

    text = bits_to_text(bits[:-8])  # Remove stop sequence
    return text

# CLI Argument Parsing
parser = argparse.ArgumentParser(description="LSB Steganography for PNG images.")
parser.add_argument("--embed", action="store_true", help="Embed text into an image")
parser.add_argument("--extract", action="store_true", help="Extract text from an image")
parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image")
parser.add_argument("-o", "--output", type=str, help="Path to save output image (only for embedding)")
parser.add_argument("-t", "--text", type=str, help="Text to embed (required for embedding)")
parser.add_argument("--mode", type=str, choices=["ordered", "random"], default="ordered", help="Mode: 'ordered' (default) or 'random'")
parser.add_argument("-a", "--a_value", type=int, default=13, help="Value of 'a' for shuffling (default: 13)")

args = parser.parse_args()

if args.embed:
    if not args.text or not args.output:
        parser.error("--embed requires --text and --output.")
    embed_text_in_png(args.input, args.output, args.text, args.mode, args.a_value)

elif args.extract:
    extracted_text = extract_text_from_png(args.input, args.mode, args.a_value)
    print("📝 Extracted text:", extracted_text)

else:
    parser.error("You must specify either --embed or --extract.")
