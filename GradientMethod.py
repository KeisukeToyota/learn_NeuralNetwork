from sympy import *

x = Symbol('x')


def gradientMax(f=x):
    x1 = 0
    e = 0.01
    g = diff(f, x)
    fold = f.subs(x, x1)
    while(True):
        x1 = x1 + e * g.subs(x, x1)
        if(f.subs(x, x1) <= fold):
            return [x1, f.subs(x, x1)]
        fold = f.subs(x, x1)


def main():
    print(gradientMax(f=-2 * x**2 + 5 * x + 2))


if __name__ == '__main__':
    main()
