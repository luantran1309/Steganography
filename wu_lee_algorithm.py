"""
-------------------- Wu-Lee Algorithm ---------------------
- Mainly used in black and white photography
- Divide the image into multiple bit blocks
- Maximum bits embedded in a block
- Extract text from embeded text

"""
from PIL import Image
import numpy as np
import argparse
import ast

def image_to_matrix(image_path):
    img = Image.open(image_path).convert("L")  # Convert image to grayscale image
    threshold = 128  # Threshold to convert to binary
    matrix = [[0 if img.getpixel((x, y)) < threshold else 1 for x in range(img.width)] for y in range(img.height)]
    return matrix

def matrix_to_image(matrix, output_path):
    matrix = np.array(matrix) * 255  # Convert 0-1 to 0-255
    img = Image.fromarray(matrix.astype(np.uint8))
    img.save(output_path)
    print("Save success " + output_path + " file")

def text_to_bits(text):
    return ''.join(f'{ord(c):08b}' for c in text)

def bits_to_text(bits):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(c, 2)) for c in chars)

def sumMatrix(matrix):
    cnt = 0
    for row in range (len(matrix)):
        for col in range(len(matrix[0])):
            if matrix[row][col] == 1:
                cnt += 1
    return cnt

def sumOf2Matrices(matrix1, matrix2):
    cnt = 0
    for row in range(len(matrix1)):
        for col in range(len(matrix1[0])):
            tmp = matrix1[row][col] & matrix2[row][col]
            if tmp == 1:
                cnt += 1
    return cnt

def extractBlock(matrix, rowCur, colCur,  length, width):
    block = [[0] * width for _ in range(length)]
    for row in range(length):
        for col in range(width):
            block[row][col] = matrix[rowCur + row][colCur + col]
    return block

def findReplace(key, block, valKey, valBlock):
    for row in range(len(key)):
        for col in range(len(key[0])):
            if(key[row][col] == valKey and block[row][col] == valBlock):
                return row, col

def findReplaceB(key, valKey):
    for row in range(len(key)):
        for col in range(len(key[0])):
            if(key[row][col] == valKey):
                return row, col

def embed(input, output, key, text):
    text = text_to_bits(text) + '00000000'  # Add 8 bits after to mark the end text
    key = ast.literal_eval(key)
    matrix = image_to_matrix(input)

    sumKey = sumMatrix(key)

    rowBlocks = len(matrix) // len(key)
    colBlocks = len(matrix[0]) // len(key[0])

    if len(text) - 8 > (rowBlocks * colBlocks):
        print("The text string is too long or the key is too large! Cannot embed enough information")
        return

    idx = 0

    for rowBlock in range(rowBlocks):
        if idx >= len(text):
            break
        for colBlock in range(colBlocks):
            if idx >= len(text):
                break
            block = extractBlock(matrix, rowBlock * len(key), colBlock* len(key[0]), len(key), len(key[0]))
            sum2Matrices = sumOf2Matrices(block, key)
            if sum2Matrices <= 0 or sum2Matrices >= sumKey:
                continue
            elif sum2Matrices % 2 == int(text[idx]):
                idx += 1
            elif sum2Matrices % 2 == 1:
                row, col  = findReplace(key, block, 1, 0)
                matrix[rowBlock * len(key) + row][colBlock * len(key[0]) + col] = 1
                idx += 1
            elif sum2Matrices == sumKey - 1:
                row, col  = findReplace(key, block, 1, 1)
                matrix[rowBlock * len(key) + row][colBlock * len(key[0]) + col] = 0
                idx += 1
            else:
                row, col = findReplaceB(key, 1)
                matrix[rowBlock * len(key) + row][colBlock * len(key[0]) + col] ^= 1
                idx += 1

    matrix_to_image(matrix, output)

def extract(input, key):
    text = ''
    key = ast.literal_eval(key)
    matrix = image_to_matrix(input)
    sumKey = sumMatrix(key)

    rowBlocks = len(matrix) // len(key)
    colBlocks = len(matrix[0]) // len(key[0])

    for rowBlock in range(rowBlocks):
        if len(text) >= 8 and text[-8:] == '00000000':
            break
        for colBlock in range(colBlocks):
            block = extractBlock(matrix, rowBlock * len(key), colBlock* len(key[0]), len(key), len(key[0]))
            sum2Matrices = sumOf2Matrices(block, key)
            if sum2Matrices <= 0 or sum2Matrices >= sumKey:
                continue
            else:
                text += str(sum2Matrices % 2)
                if len(text) >= 8 and text[-8:] == '00000000':
                    break
    print("📝 Extracted text:", bits_to_text(text))


# CLI Argument Parsing
parser = argparse.ArgumentParser(description="Wu-Lee algorithm to embed text into image.")
parser.add_argument("--embed", action="store_true", help="Embed text into an image")
parser.add_argument("--extract", action="store_true", help="Extract text from an image")
parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image")
parser.add_argument("-o", "--output", type=str, help="Path to save output image (only for embedding)")
parser.add_argument("-t", "--text", type=str, help="Text to embed (required for embedding)")
parser.add_argument("-k", "--key", type=str, required=True, help="required for embedding and extracting. Ex: [[1,1,0],[0,1,1]]")

args = parser.parse_args()

if args.embed:
    if not args.text or not args.key or not args.output:
        parser.error("--embed requires --text, --key and --output.")
    embed(args.input, args.output, args.key, args.text)

elif args.extract:
    if not args.key:
        parser.error("--extract requires --key.")
    extract(args.input, args.key)

else:
    parser.error("You must specify either --embed or --extract.")
