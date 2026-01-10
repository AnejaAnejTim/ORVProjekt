import os
import time
import math
import collections
import struct
import numpy as np

class BitStream:
    def __init__(self):
        self.bits = collections.deque()

    def append_bits(self, value, num_bits):
        if num_bits == 0:
            return
        for i in range(num_bits - 1, -1, -1):
            self.bits.append((value >> i) & 1)

    def get_bits(self, num_bits):
        if num_bits == 0:
            return 0
        value = 0
        for _ in range(num_bits):
            if not self.bits:
                raise ValueError("No more bits")
            bit = self.bits.popleft()
            value = (value << 1) | bit
        return value

    def to_bytes(self):
        num_bytes = (len(self.bits) + 7) // 8
        data = bytearray(num_bytes)
        for i, bit in enumerate(self.bits):
            byte_idx = i // 8
            bit_idx = 7 - (i % 8)
            data[byte_idx] |= bit << bit_idx
        return data

    @classmethod
    def from_bytes(cls, data):
        bs = cls()
        for byte in data:
            for i in range(7, -1, -1):
                bit = (byte >> i) & 1
                bs.bits.append(bit)
        return bs

def decode_header(bs):
    height = bs.get_bits(16)
    c0 = bs.get_bits(8)
    cn1 = bs.get_bits(32)
    n = bs.get_bits(32)
    return height, n, c0, cn1

def deic(bs, C, L, H):
    if H - L > 1:
        cl = C[L]
        ch = C[H]
        if cl == ch:
            for i in range(L + 1, H):
                C[i] = cl
        else:
            m = (H + L) // 2
            diff = ch - cl + 1
            g = math.ceil(math.log2(diff))
            val = bs.get_bits(g)
            C[m] = cl + val
            if L < m:
                deic(bs, C, L, m)
            if m < H:
                deic(bs, C, m, H)

def predict_inverse(epsilon, height, width):
    P = [[0 for _ in range(width)] for _ in range(height)]
    for x in range(width):
        for y in range(height):
            i = y + x * height
            if x == 0 and y == 0:
                P[y][x] = epsilon[i]
            elif x == 0:
                pred = P[y - 1][x]
                P[y][x] = pred - epsilon[i]
            elif y == 0:
                pred = P[y][x - 1]
                P[y][x] = pred - epsilon[i]
            else:
                a = P[y][x - 1]
                b = P[y - 1][x]
                c = P[y - 1][x - 1]
                if c >= max(a, b):
                    pred = min(a, b)
                elif c <= min(a, b):
                    pred = max(a, b)
                else:
                    pred = a + b - c
                P[y][x] = pred - epsilon[i]
    return P

def decompress(bs):
    height, n, c0, cn1 = decode_header(bs)
    width = n // height
    C = [0] * n
    C[0] = c0
    C[n - 1] = cn1
    deic(bs, C, 0, n - 1)
    N = [0] * n
    N[0] = C[0]
    for i in range(1, n):
        N[i] = C[i] - C[i - 1]
    epsilon = [0] * n
    epsilon[0] = N[0]
    for i in range(1, n):
        if N[i] % 2 == 0:
            epsilon[i] = N[i] // 2
        else:
            epsilon[i] = - (N[i] + 1) // 2
    P = predict_inverse(epsilon, height, width)
    return P

def bmp_write_24bit(path, R, G, B):
    height = len(R)
    width = len(R[0]) if height > 0 else 0
    row_stride = ((width * 3 + 3) // 4) * 4
    pixel_array_size = row_stride * height
    biSize = 40
    bfOffBits = 14 + biSize
    bfSize = bfOffBits + pixel_array_size
    header = bytearray()
    header += b'BM'
    header += struct.pack('<IHHI', bfSize, 0, 0, bfOffBits)
    header += struct.pack('<I', biSize)
    header += struct.pack('<ii', width, height)
    header += struct.pack('<HH', 1, 24)
    header += struct.pack('<I', 0)
    header += struct.pack('<I', pixel_array_size)
    header += struct.pack('<ii', 2835, 2835)
    header += struct.pack('<II', 0, 0)
    
    pixels = bytearray(pixel_array_size)
    for row in range(height):
        dst_row_index = height - 1 - row
        row_start = dst_row_index * row_stride
        for x in range(width):
            # BMP stores in BGR order
            for i, channel in enumerate([B, G, R]):
                val = channel[row][x]
                if val < 0: val = 0
                elif val > 255: val = 255
                pixels[row_start + x * 3 + i] = int(val)
                
    with open(path, 'wb') as f:
        f.write(header)
        f.write(pixels)

def decompress_file(input_file, output_file):
    with open(input_file, 'rb') as f:
        data = f.read()
    bs = BitStream.from_bytes(data)
    
    # Decompress three channels (R, G, B)
    R = decompress(bs)
    G = decompress(bs)
    B = decompress(bs)
    
    bmp_write_24bit(output_file, R, G, B)
