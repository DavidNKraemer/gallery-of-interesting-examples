from collections import defaultdict
import numpy as np

def first_digit_doubling():
    current = 1
    frequencies = np.zeros(10)
    count = 1
    while True:
        first_digit = int(str(current)[0])
        frequencies[first_digit] += 1
        current *= 2
        count += 1
        yield first_digit, frequencies, count

max_iter = 10000
generator = first_digit_doubling()
for i in range(max_iter):
    _ = next(generator)

digit, frequencies, count = next(generator)
print(frequencies / count)



