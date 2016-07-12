"""
確率的2値モデル
"""
import random
import math
import sys

param = sys.argv
gain = 1
threshold = 1.0
loop = int(param[1])
input1 = [1, 0, 1]
input2 = [1, -1, 0]
weight = [3, 2, -1]


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-gain * z))


def potential(n, w, x):
    s = 0
    for i in range(n):
        s += w[i] * x[i]
    return s


def theoretical_value(w, input):
    return int(loop * sigmoid(potential(3, w, input) - threshold))


def measured_value(w, input):
    count = 0
    for i in range(loop):
        if random.random() <= sigmoid(potential(3, w, input) - threshold):
            count += 1
    return count


def main():
    print ('実測値１：%d/%d' % (measured_value(weight, input1), loop))
    print ('理論値１：%d/%d' % (theoretical_value(weight, input1), loop))
    print ('実測値２：%d/%d' % (measured_value(weight, input2), loop))
    print ('理論値２：%d/%d' % (theoretical_value(weight, input2), loop))


if __name__ == "__main__":
    main()
