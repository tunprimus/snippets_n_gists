#!/usr/bin/env python3
""" Still non-functional """

# Adapted and extended from
# How to Implement a Random String Generator With Python
# https://miguendes.me/how-to-implement-a-random-string-generator-with-python

import string
import random

str_alphabets = string.ascii_letters

str_nums = string.digits

str_special_chars = string.punctuation

def main():
    gen_random_chars()

def gen_random_chars(length = 16, alphabet = True, num = True, special = True):
    if alphabet and num and special:
        all_chars = str_alphabets + str_nums + str_special_chars
    return "".join((random.SystemRandom.choices(all_chars, k = length)))


if __name__ == "__main__":
    main()