# CRC + Extended Hamming (SECDED) simulation
# Jupyter-friendly Python script
# Author: ChatGPT
# Usage: run cells in order in a Jupyter notebook (or import functions)

import random
import math
import numpy as np
from collections import namedtuple

# ---------------------------
# Utilities: bit conversions
# ---------------------------

def bytes_to_bits(data_bytes):
    bits = []
    for b in data_bytes:
        for i in range(8)[::-1]:
            bits.append((b >> i) & 1)
    return bits

def bits_to_bytes(bits):
    # pad to full byte
    pad = (-len(bits)) % 8
    bits = bits + [0]*pad
    out = bytearray()
    for i in range(0, len(bits), 8):
        val = 0
        for j in range(8):
            val = (val << 1) | bits[i+j]
        out.append(val)
    return bytes(out)

# ---------------------------
# Simple CRC implementation (bitwise)
# generator polynomial provided as int (e.g., 0x1021 for CRC-16-CCITT)
# ---------------------------

def int_to_poly_bits(poly_int):
    # returns list of bits of polynomial from MSB to LSB excluding leading 1
    deg = poly_int.bit_length() - 1
    return [(poly_int >> i) & 1 for i in range(deg-1, -1, -1)]


def crc_append(data_bits, poly_int, crc_len=None):
    '''Append CRC to data bits using polynomial poly_int.
       If crc_len is None it uses poly_int bit-length - 1.'''
    poly_deg = poly_int.bit_length() - 1
    if crc_len is None:
        crc_len = poly_deg
    # make a copy and append zeros
    msg = data_bits[:] + [0]*crc_len
    poly = poly_int
    for i in range(len(data_bits)):
        if msg[i] == 1:
            # XOR with poly shifted
            for j in range(poly_deg+1):
                msg[i+j] ^= (poly >> (poly_deg - j)) & 1
    crc = msg[-crc_len:]
    return data_bits + crc


def crc_check(received_bits, poly_int, crc_len=None):
    poly_deg = poly_int.bit_length() - 1
    if crc_len is None:
        crc_len = poly_deg
    msg = received_bits[:]
    for i in range(len(msg) - crc_len):
        if msg[i] == 1:
            for j in range(poly_deg+1):
                msg[i+j] ^= (poly_int >> (poly_deg - j)) & 1
    # remainder should be all zeros
    return any(bit == 1 for bit in msg[-crc_len:]) == False

# ---------------------------
# Extended Hamming (SECDED) - generic implementation
# For any k bits of data, compute number of parity bits r satisfying 2^r >= k + r + 1
# Extended adds one overall parity bit for double-error detection
# Positions are 1-indexed for parity calculations
# ---------------------------

HammingResult = namedtuple('HammingResult', ['corrected_data', 'status', 'corrected_codeword', 'syndrome', 'overall_parity'])


def calc_r_for_k(k):
    r = 1
    while 2**r < (k + r + 1):
        r += 1
    return r


def insert_parity_positions(data_bits):
    k = len(data_bits)
    r = calc_r_for_k(k)
    n = k + r + 1  # +1 for overall parity
    code = [None] * n
    # place data bits into codeword (1-indexed positions)
    j = 0
    for i in range(1, n+1):
        if (i & (i-1)) == 0:  # power of two -> parity position
            code[i-1] = 0
        else:
            code[i-1] = data_bits[j]
            j += 1
    return code, r


def compute_parity_bits(code, r):
    n = len(code)
    # compute parity bits at positions 1,2,4,... (1-indexed)
    for i in range(r):
        pos = 2**i
        parity = 0
        for j in range(1, n+1):
            if j & pos:
                if code[j-1] is None:
                    continue
                parity ^= code[j-1]
        code[pos-1] = parity
    # overall parity (even parity over entire codeword)
    overall = 0
    for b in code:
        overall ^= (0 if b is None else b)
    code.append(overall)
    return code


def encode_hamming(data_bits):
    code, r = insert_parity_positions(data_bits)
    code = compute_parity_bits(code, r)
    return code


def decode_hamming(codeword):
    # codeword is list of bits, last bit is overall parity
    n_total = len(codeword)
    overall = codeword[-1]
    code = codeword[:-1]  # without overall
    n = len(code)
    # determine r (number of parity bits). find largest power of two <= n
    r = 0
    while 2**r <= n:
        r += 1
    r = r  # r parity positions exist
    syndrome = 0
    for i in range(r):
        pos = 2**i
        parity = 0
        for j in range(1, n+1):
            if j & pos:
                parity ^= code[j-1]
        if parity:
            syndrome += pos
    # overall parity check (including overall bit)
    total_parity = overall
    for b in code:
        total_parity ^= b
    status = 'no_error'
    corrected = code[:]  # copy
    if syndrome == 0 and total_parity == 0:
        status = 'no_error'
    elif syndrome == 0 and total_parity == 1:
        # error in overall parity bit itself -> correct it
        status = 'corrected_overall_parity'
        overall ^= 1
    elif syndrome != 0 and total_parity == 1:
        # single-bit error at position syndrome -> correct
        status = 'corrected_single_bit'
        if 1 <= syndrome <= n:
            corrected[syndrome-1] ^= 1
        else:
            # syndrome points to parity outside normal range -> treat as uncorrectable
            status = 'uncorrectable'
    elif syndrome != 0 and total_parity == 0:
        # detected double error
        status = 'detected_double_error'
    # extract data bits from corrected
    data_bits = []
    for i in range(1, n+1):
        if (i & (i-1)) != 0:
            data_bits.append(corrected[i-1])
    return HammingResult(corrected_data=data_bits, status=status, corrected_codeword=corrected + [overall], syndrome=syndrome, overall_parity=overall)

# ---------------------------
# Channel simulation: flip random bits
# ---------------------------

def flip_bits(bitlist, error_prob):
    out = bitlist[:]
    for i in range(len(out)):
        if random.random() < error_prob:
            out[i] ^= 1
    return out

# ---------------------------
# High-level pipeline
# data -> append CRC -> split into hamming-blocks (k bits each) -> encode each -> send -> decode each -> reassemble -> check CRC
# ---------------------------

def split_bits(bits, block_k):
    # pad last block with zeros if necessary
    pad = (-len(bits)) % block_k
    if pad:
        bits = bits + [0]*pad
    blocks = [bits[i:i+block_k] for i in range(0, len(bits), block_k)]
    return blocks


def encode_message_with_crc_and_hamming(data_bytes, crc_poly=0x1021, hamming_k=11):
    data_bits = bytes_to_bits(data_bytes)
    # append CRC
    msg_with_crc = crc_append(data_bits, crc_poly)
    # split into data blocks of length hamming_k
    blocks = split_bits(msg_with_crc, hamming_k)
    encoded_blocks = []
    for blk in blocks:
        code = encode_hamming(blk)
        encoded_blocks.extend(code)
    return encoded_blocks, len(msg_with_crc)


def decode_message_with_crc_and_hamming(received_bits, msg_with_crc_len, hamming_k=11):
    # decode blocks
    r = calc_r_for_k(hamming_k)
    n = hamming_k + r + 1
    blocks = [received_bits[i:i+n] for i in range(0, len(received_bits), n)]
    recovered = []
    stats = {'corrected_single':0, 'corrected_overall':0, 'double_errors':0, 'uncorrectable':0}
    for b in blocks:
        res = decode_hamming(b)
        if res.status == 'corrected_single_bit':
            stats['corrected_single'] += 1
        elif res.status == 'corrected_overall_parity':
            stats['corrected_overall'] += 1
        elif res.status == 'detected_double_error':
            stats['double_errors'] += 1
        elif res.status == 'uncorrectable':
            stats['uncorrectable'] += 1
        recovered.extend(res.corrected_data)
    # trim to original message+crc length
    recovered = recovered[:msg_with_crc_len]
    crc_ok = crc_check(recovered, 0x1021)
    # separate original data and crc bits
    k_crc = 0x1021.bit_length() - 1
    data_bits = recovered[:-k_crc]
    return bits_to_bytes(data_bits), crc_ok, stats

# ---------------------------
# Quick demo / simulation
# ---------------------------
if __name__ == '__main__':
    random.seed(42)
    # choose message
    message = b'Hello CRC+Hamming'
    print('Original message:', message)
    encoded, msg_with_crc_len = encode_message_with_crc_and_hamming(message, crc_poly=0x1021, hamming_k=11)
    # simulate channel with bit error rate
    ber = 0.001  # bit error rate
    received = flip_bits(encoded, ber)
    decoded_msg, crc_ok, stats = decode_message_with_crc_and_hamming(received, msg_with_crc_len, hamming_k=11)
    print('Decoded message:', decoded_msg)
    print('CRC OK after Hamming decode?:', crc_ok)
    print('Hamming stats:', stats)

    # run many trials to estimate end-to-end detection/correction
    trials = 2000
    ber = 0.002
    detected_failures = 0
    undetected_failures = 0
    for t in range(trials):
        encoded, msg_with_crc_len = encode_message_with_crc_and_hamming(message, crc_poly=0x1021, hamming_k=11)
        received = flip_bits(encoded, ber)
        decoded_msg, crc_ok, stats = decode_message_with_crc_and_hamming(received, msg_with_crc_len, hamming_k=11)
        if not crc_ok:
            # CRC failed -> either detected or undetected? crc_ok False means CRC detected error
            detected_failures += 1
        if decoded_msg != message:
            undetected_failures += 1
    print('\nSimulation over', trials, 'trials with BER=', ber)
    print('CRC detected errors (after Hamming):', detected_failures)
    print('Decoded messages not equal to original:', undetected_failures)
    print('Note: decoded != original can be due to Hamming uncorrectable or CRC passing despite errors (rare).')
