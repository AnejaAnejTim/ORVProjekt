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

    def copy(self):
        bs = BitStream()
        bs.bits = collections.deque(self.bits)
        return bs

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


def set_header(bs, height, c0, cn1, n):
    bs.append_bits(height, 16)  # visina
    bs.append_bits(c0, 8)  # prvi C
    bs.append_bits(cn1, 32)  # zadnji iz C
    bs.append_bits(n, 32)  # st vseh el.


def decode_header(bs):
    height = bs.get_bits(16)
    c0 = bs.get_bits(8)
    cn1 = bs.get_bits(32)
    n = bs.get_bits(32)
    return height, n, c0, cn1


def ic(bs, C, L, H):
    if H - L > 1:
        cl = C[L]
        ch = C[H]
        if ch != cl:
            m = (H + L) // 2
            diff = ch - cl + 1
            g = math.ceil(math.log2(diff))
            val = C[m] - cl
            bs.append_bits(val, g)
            if L < m:
                ic(bs, C, L, m)
            if m < H:
                ic(bs, C, m, H)


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


def predict(P, height, width):
    """ napoved JPG-LS """
    epsilon = [0] * (height * width)
    for x in range(width):
        for y in range(height):
            i = y + x * height  # E[y*X+x]
            if x == 0 and y == 0:
                epsilon[i] = P[0][0]
            # • Če je y = 0
            elif y == 0:
                # • E[y*X + x] = P[x - 1, 0] - P[x, 0]
                pred = P[0][x - 1]
                epsilon[i] = pred - P[0][x]
            # • Če je x = 0
            elif x == 0:
                # • E[y*X + x] = P[0, y - 1] - P[0, y]
                pred = P[y - 1][0]
                epsilon[i] = pred - P[y][0]
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
                epsilon[i] = pred - P[y][x]
    return epsilon


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


def compress(P, height, width):
    epsilon = predict(P, height, width)
    n = height * width
    N = [0] * n
    N[0] = epsilon[0]
    for i in range(1, n):
        if epsilon[i] >= 0:
            N[i] = 2 * epsilon[i]
        else:
            N[i] = 2 * abs(epsilon[i]) - 1

    C = [0] * n
    C[0] = N[0]
    for i in range(1, n):
        C[i] = C[i - 1] + N[i]
    bs = BitStream()
    set_header(bs, height, C[0], C[-1], n)
    ic(bs, C, 0, n - 1)
    return bs


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


def decompress_block(block_data):
    bs = BitStream.from_bytes(block_data)
    return decompress(bs)


def decompress_parallel(data):
    from mpi4py import MPI
    import subprocess
    import sys

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size <= 1:
        # Check if we should try to run with mpiexec
        # This is a heuristic: if we are in Rank 0 and Size 1, and we see a .bin file,
        # we might want to start the MPI cluster if not already running.
        # However, to avoid infinite recursion and port conflicts, we only do this
        # if explicitly requested or via a specific entry point.
        # For now, we will just implement the fallback and provide a way to trigger the script.

        # Fallback to single-process if not running in MPI environment
        bs_global = BitStream.from_bytes(data)
        height = bs_global.get_bits(16)
        width = bs_global.get_bits(16)
        block_size = bs_global.get_bits(16)

        num_blocks_per_channel = ((height + block_size - 1) // block_size) * \
                                 ((width + block_size - 1) // block_size)
        total_blocks = 3 * num_blocks_per_channel

        block_sizes = []
        for _ in range(total_blocks):
            block_sizes.append(bs_global.get_bits(32))

        header_bits_total = 16 * 3 + 32 * total_blocks
        header_bytes_total = (header_bits_total + 7) // 8
        
        block_data_start = header_bytes_total
        results = []
        curr_offset = block_data_start
        for b_size in block_sizes:
            results.append(decompress_block(data[curr_offset:curr_offset + b_size]))
            curr_offset += b_size
    else:
        if rank == 0:
            bs_global = BitStream.from_bytes(data)
            height = bs_global.get_bits(16)
            width = bs_global.get_bits(16)
            block_size = bs_global.get_bits(16)

            num_blocks_per_channel = ((height + block_size - 1) // block_size) * \
                                     ((width + block_size - 1) // block_size)
            total_blocks = 3 * num_blocks_per_channel

            block_sizes = []
            for _ in range(total_blocks):
                block_sizes.append(bs_global.get_bits(32))

            header_bits_total = 16 * 3 + 32 * total_blocks
            header_bytes_total = (header_bits_total + 7) // 8
            
            block_data_start = header_bytes_total
            all_blocks_data = []
            curr_offset = block_data_start
            for b_size in block_sizes:
                all_blocks_data.append(data[curr_offset:curr_offset + b_size])
                curr_offset += b_size

            results = [None] * total_blocks
            for w in range(1, size):
                comm.send("START", dest=w, tag=0)

            blocks_sent_to_workers = 0
            for i, block_payload in enumerate(all_blocks_data):
                target_rank = i % size
                if target_rank == 0:
                    results[i] = decompress_block(block_payload)
                else:
                    comm.send((i, block_payload), dest=target_rank, tag=1)
                    blocks_sent_to_workers += 1

            for w in range(1, size):
                comm.send(None, dest=w, tag=1)

            for _ in range(blocks_sent_to_workers):
                idx, block_p = comm.recv(source=MPI.ANY_SOURCE, tag=2)
                results[idx] = block_p
        else:
            return None

    if rank == 0:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        blocks_in_row = (width + block_size - 1) // block_size
        
        for i, block_p in enumerate(results):
            if block_p is None: continue
            channel_idx = i // num_blocks_per_channel
            block_idx_in_channel = i % num_blocks_per_channel
            
            br = (block_idx_in_channel // blocks_in_row) * block_size
            bc = (block_idx_in_channel % blocks_in_row) * block_size
            
            bh = len(block_p)
            bw = len(block_p[0])
            
            for r in range(bh):
                for c in range(bw):
                    img[br + r, bc + c, 2 - channel_idx] = block_p[r][c]
        
        return img
    return None

def mpi_worker_loop():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    print(f"Worker {rank} started and waiting for tasks.")
    
    while True:
        signal = comm.recv(source=0, tag=0)
        if signal == "EXIT":
            break
        
        while True:
            task = comm.recv(source=0, tag=1)
            if task is None: # End of current image blocks
                break
            
            idx, block_payload = task
            try:
                block_p = decompress_block(block_payload)
                comm.send((idx, block_p), dest=0, tag=2)
            except Exception as e:
                print(f"Worker {rank} error on block {idx}: {e}")
                comm.send((idx, None), dest=0, tag=2)


def bmp_read_8bit(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data[0:2] != b'BM':
        raise ValueError('Not a BMP file')
    bfSize, bfReserved1, bfReserved2, bfOffBits = struct.unpack_from('<IHHI', data, 2)
    biSize = struct.unpack_from('<I', data, 14)[0]
    if biSize != 40:
        raise ValueError('Only BITMAPINFOHEADER (40 bytes) supported')
    (biWidth, biHeight, biPlanes, biBitCount, biCompression,
     biSizeImage, biXPelsPerMeter, biYPelsPerMeter,
     biClrUsed, biClrImportant) = struct.unpack_from('<iiHHIIIIII', data, 18)
    if biBitCount != 8:
        raise ValueError('Only 8-bit BMP supported')
    if biCompression != 0:
        raise ValueError('Only uncompressed BI_RGB BMP supported')
    width = biWidth
    height = abs(biHeight)
    palette_size = (biClrUsed if biClrUsed else 256) * 4
    palette_offset = 14 + biSize
    pixel_offset = bfOffBits
    row_stride = ((width + 3) // 4) * 4
    P = [[0 for _ in range(width)] for _ in range(height)]
    bottom_up = (biHeight > 0)
    for row in range(height):
        src_row_index = (height - 1 - row) if bottom_up else row
        row_start = pixel_offset + src_row_index * row_stride
        row_bytes = data[row_start: row_start + row_stride]
        for x in range(width):
            P[row][x] = row_bytes[x]
    return P, height, width


def bmp_write_8bit(path, P):
    height = len(P)
    width = len(P[0]) if height > 0 else 0
    row_stride = ((width + 3) // 4) * 4
    pixel_array_size = row_stride * height
    biSize = 40
    bfOffBits = 14 + biSize + 256 * 4
    bfSize = bfOffBits + pixel_array_size
    header = bytearray()
    header += b'BM'
    header += struct.pack('<IHHI', bfSize, 0, 0, bfOffBits)
    header += struct.pack('<I', biSize)
    header += struct.pack('<ii', width, height)
    header += struct.pack('<HH', 1, 8)
    header += struct.pack('<I', 0)
    header += struct.pack('<I', pixel_array_size)
    header += struct.pack('<ii', 2835, 2835)
    header += struct.pack('<II', 256, 0)
    palette = bytearray()
    for i in range(256):
        palette += bytes((i, i, i, 0))
    pixels = bytearray(pixel_array_size)
    for row in range(height):
        dst_row_index = height - 1 - row
        row_start = dst_row_index * row_stride
        for x in range(width):
            val = P[row][x]
            if val < 0:
                val = 0
            elif val > 255:
                val = 255
            pixels[row_start + x] = val
    with open(path, 'wb') as f:
        f.write(header)
        f.write(palette)
        f.write(pixels)


def compress_image(input_file, output_file):
    P, height, width = bmp_read_8bit(input_file)
    bs = compress(P, height, width)
    data = bs.to_bytes()
    with open(output_file, 'wb') as f:
        f.write(data)


def decompress_image(input_file, output_file):
    with open(input_file, 'rb') as f:
        data = f.read()
    bs = BitStream.from_bytes(data)
    P = decompress(bs)
    bmp_write_8bit(output_file, P)


def process_all_from_slikeBMP():
    src_dir = os.path.join(os.path.dirname(__file__), 'slikeBMP')
    if not os.path.isdir(src_dir):
        print(f'Not found: {src_dir}')
        return
    out_dir = os.path.join(src_dir, 'out')
    os.makedirs(out_dir, exist_ok=True)

    entries = sorted([f for f in os.listdir(src_dir) if f.lower().endswith('.bmp')])
    if not entries:
        print('No BMP files found in slikeBMP')
        return

    total = len(entries)
    entries = entries[:10]
    print('Processing BMP images from slikeBMP:')
    print(f'Total found: {total} files; processing first {len(entries)}')
    print()

    for fname in entries:
        bmp_path = os.path.join(src_dir, fname)
        base, _ = os.path.splitext(fname)
        comp_path = os.path.join(out_dir, base + '.flocic')
        dec_path = os.path.join(out_dir, base + '_dec.bmp')

        try:
            t0 = time.perf_counter()
            compress_image(bmp_path, comp_path)
            t1 = time.perf_counter()
            decompress_image(comp_path, dec_path)
            t2 = time.perf_counter()

            orig_size = os.path.getsize(bmp_path)
            comp_size = os.path.getsize(comp_path)
            ratio = (orig_size / comp_size) if comp_size > 0 else float('inf')

            P1, h1, w1 = bmp_read_8bit(bmp_path)
            P2, h2, w2 = bmp_read_8bit(dec_path)
            same = (h1 == h2 and w1 == w2 and all(P1[y][x] == P2[y][x] for y in range(h1) for x in range(w1)))

            print(f'{fname}: orig={orig_size}B, comp={comp_size}B, ratio={ratio:.3f}, ' \
                  f'comp_time={(t1 - t0) * 1000:.1f}ms, decomp_time={(t2 - t1) * 1000:.1f}ms, match={same}')
        except Exception as e:
            print(f'Error processing {fname}: {e}')


if __name__ == '__main__':
    process_all_from_slikeBMP()
