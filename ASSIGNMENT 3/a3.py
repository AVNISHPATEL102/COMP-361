#!/usr/bin/env python

from numpy import *
from math import *

'''
NOTE: You are not allowed to import any function from numpy's linear 
algebra library, or from any other library except math.
'''

'''
    Part 1: Warm-up (bonus point)
'''


def python2_vs_python3():
    '''
    A few of you lost all their marks in A2 because their assignment contained
    Python 2 code that does not work in Python 3, in particular print statements
    without parentheses. For instance, 'print hello' is valid in Python 2 but not
    in Python 3.
    Remember that you are strongly encouraged to check the outcome of the tests
    by running pytest on your computer **with Python 3** and by checking Travis.
    Task: Nothing to implement in this function, that's a bonus point, yay!
          Just don't loose it by adding Python 2 syntax to this file...
    Test: 'tests/test_python3.py'
    '''
    return ("I won't use Python 2 syntax my code",
            "I will always use parentheses around print statements ",
            "I will check the outcome of the tests using pytest or Travis"
            )


'''
    Part 2: Integration (Chapter 6)
'''


def problem_6_1_18(x):
    '''
    We will solve problem 6.1.18 in the textbook.
    Task: The function must return the integral of sin(t)/t 
          between 0 and x:
              problem_6_1_18(x) = int_0^x{sin(t)/t dt}
    Example: problem_6_1_18(1.0) = 0.94608
    Test: Function 'test_problem_6_1_18' in 'tests/test_problem_6_1_18.py'
    Hints: * use the composite trapezoid rule
           * the integrated function has a singularity in 0. An easy way
             to deal with this is to integrate between a small positive value and x.
    '''

    ## YOUR CODE HERE
    def f(x):
        return sin(x) / x

    #     print(trapezoid(f, 0.10e-5, x, 1000))
    return trapezoid(f, 0.0001, x, 100)
    raise Exception("Not implemented")


def example_6_12():
    '''
    We will implement example 6.12 in the textbook:
        "
            Evaluate the value of int_1.5^3 f(x)dx ('the integral of f(x)
            between 1.5 and 3'), where f(x) is represented by the
            unevenly spaced data points defined in x_data and y_data.
        "
    Task: This function must return the value of int_1.5^3 f(x)dx where
          f(x) is represented by the evenly spaced data points in x_data and
          y_data below.
    Test: function 'test_example_6_12' in 'tests/test_example_6_12.py'.
    Hints: 1. interpolate the given points by a polynomial of degree 5.
           2. use 3-node Gauss-Legendre integration (with change of variable)
              to integrate the polynomial.
    '''

    x_data = array([1.2, 1.7, 2.0, 2.4, 2.9, 3.3])
    y_data = array([-0.36236, 0.12884, 0.41615, 0.73739, 0.97096, 0.98748])

    coeff = interpolat(x_data, y_data)

    def f(x):
        return coeff[0] + coeff[1] * x + coeff[2] * x ** 2 + coeff[3] * x ** 3 + coeff[4] * x ** 4 + coeff[5] * x ** 5

    #     print(trapezoid(f, 1.5, 3, 1000))
    return trapezoid(f, 1.5, 3, 100)

    ## YOUR CODE HERE
    raise Exception("Not implemented")


'''
    Part 3: Initial-Value Problems
'''


def problem_7_1_8(x):
    '''
    We will solve problem 7.1.8 in the textbook. A skydiver of mass m in a
    vertical free fall experiences an aerodynamic drag force F=cy'² ('c times
    y prime square') where y is measured downward from the start of the fall,
    and y is a function of time (y' denotes the derivative of y w.r.t time).
    The differential equation describing the fall is:
         y''=g-(c/m)y'²
    And y(0)=y'(0)=0 as this is a free fall.
    Task: The function must return the time of a fall of x meters, where
          x is the parameter of the function. The values of g, c and m are
          given below.
    Test: function 'test_problem_7_1_8' in 'tests/test_problem_7_1_8.py'
    Hint: use Runge-Kutta 4.
    '''
    g = 9.80665  # m/s**2
    c = 0.2028  # kg/m
    m = 80  # kg

    ## YOUR CODE HERE
    def F(x, y):
        return array([
            y[1],
            g - ((c / m) * (y[1] ** 2))
        ])

    y0 = array([
        0,
        0
    ])
    X, Y = runge_kutta_4(F, 0, y0, x, 10e-5)
    #     print(X[-1])
    return X[-1]
    # print(Y)
    raise Exception("Not implemented")


def problem_7_1_11(x):
    '''
    We will solve problem 7.1.11 in the textbook.
    Task: this function must return the value of y(x) where y is solution of the
          following initial-value problem:
            y' = sin(xy), y(0) = 2
    Test: function 'test_problem_7_1_11' in 'test/test_problem_7_1_11.py'
    Hint: Use Runge-Kutta 4.
    '''

    ## YOUR CODE HERE
    def F(x, y):
        return sin(x * y)

    X, Y = runge_kutta_4_2(F, 0, 2, x, 0.0001)
    #     print(Y[-1])
    return Y[-1]
    raise Exception("Not implemented")


'''
    Part 4: Two-Point Boundary Value Problems
'''


def problem_8_2_18(a, r0):
    '''
    We will solve problem 8.2.18 in the textbook. A thick cylinder of
    radius 'a' conveys a fluid with a temperature of 0 degrees Celsius in
    an inner cylinder of radius 'a/2'. At the same time, the outer cylinder is
    immersed in a bath that is kept at 200 Celsius. The goal is to determine the
    temperature profile through the thickness of the cylinder, knowing that
    it is governed by the following differential equation:
        d²T/dr²  = -1/r*dT/dr
        with the following boundary conditions:
            T(r=a/2) = 0
            T(r=a) = 200
    Task: The function must return the value of the temperature T at r=r0
          for a cylinder of radius a (a/2<=r0<=a).
    Test:  Function 'test_problem_8_2_18' in 'tests/test_problem_8_2_18'
    Hints: Use the shooting method. In the shooting method, use h=0.01
           in Runge-Kutta 4.
    '''

    ## YOUR CODE HERe
    def F(x, y):
        return array([
            y[1],
            -y[1] / x
        ])

    a0 = a * 1 / 2
    alpha = 0
    b0 = a * 1
    beta = 200
    X, Y = shooting_o2(F, a0, alpha, b0, beta, r0, 5, 50)
    return Y[-1, 0]

    raise Exception("Not implemented")


def gauss_pivot(a, b):
    gauss_elimination_pivot(a, b)
    return gauss_substitution(a, b)


def interpolat(x_data, y_data):
    A = zeros((len(x_data), len(y_data)))
    B = []
    for i in range(len(x_data)):
        # x, y = points[i]
        B.insert(i, y_data[i])
        power = 0
        for j in range(len(y_data)):
            A[i, j:j + 1] = [x_data[i] ** power]
            power = power + 1
    return gauss_pivot(A, B)


def trapezoid(f, a, b, n):
    '''
    Integrates f between a and b using n panels (n+1 points)
    '''
    h = (b - a) / n
    x = a + h * arange(n + 1)
    I = f(x[0]) / 2
    for i in range(1, n):
        I += f(x[i])
    I += f(x[n]) / 2
    return h * I


def gauss_substitution(a, b):
    n, m = shape(a)
    n2, = shape(b)
    assert (n == n2)
    x = zeros(n)
    for i in range(n - 1, -1, -1):  # decreasing index
        x[i] = (b[i] - dot(a[i, i + 1:], x[i + 1:])) / a[i, i]
    return x


def gauss_elimination_pivot(a, b, verbose=False):
    n, m = shape(a)
    n2, = shape(b)
    assert (n == n2)
    # New in pivot version
    s = zeros(n)
    for i in range(n):
        s[i] = max(abs(a[i, :]))
    for k in range(n - 1):
        # New in pivot version
        p = argmax(abs(a[k:, k]) / s[k:]) + k
        swap(a, p, k)
        swap(b, p, k)
        swap(s, p, k)
        # The remainder remains as in the previous version
        for i in range(k + 1, n):
            assert (a[k, k] != 0)  # this shouldn't happen now, unless the matrix is singular
            if (a[i, k] != 0):  # no need to do anything when lambda is 0
                lmbda = a[i, k] / a[k, k]  # lambda is a reserved keyword in Python
                a[i, k:n] = a[i, k:n] - lmbda * a[k, k:n]  # list slice operations
                b[i] = b[i] - lmbda * b[k]
            if verbose:
                print(a, b)


def swap(a, i, j):
    if len(shape(a)) == 1:
        a[i], a[j] = a[j], a[i]  # unpacking
    else:
        a[[i, j], :] = a[[j, i], :]


def runge_kutta_4(F, x0, y0, x, h):
    X = []
    Y = []
    X.append(x0)
    Y.append(y0)
    while y0[0] < x:
        k0 = F(x0, y0)
        k1 = F(x0 + h / 2.0, y0 + h / 2.0 * k0)
        k2 = F(x0 + h / 2.0, y0 + h / 2 * k1)
        k3 = F(x0 + h, y0 + h * k2)
        y0 = y0 + h / 6.0 * (k0 + 2 * k1 + 2.0 * k2 + k3)
        x0 += h
        X.append(x0)
        Y.append(y0)
    return array(X), array(Y)


def runge_kutta_4_2(F, x0, y0, x, h):
    X = []
    Y = []
    X.append(x0)
    Y.append(y0)
    while x0 < x:
        k0 = F(x0, y0)
        k1 = F(x0 + h / 2.0, y0 + h / 2.0 * k0)
        k2 = F(x0 + h / 2.0, y0 + h / 2 * k1)
        k3 = F(x0 + h, y0 + h * k2)
        y0 = y0 + h / 6.0 * (k0 + 2 * k1 + 2.0 * k2 + k3)
        x0 += h
        X.append(x0)
        Y.append(y0)
    return array(X), array(Y)


def false_position(f, a, b, delta_x):
    fa = f(a)
    fb = f(b)
    if sign(fa) == sign(fb):
        raise Exception("Root hasn't been bracketed")
    estimates = []
    while True:
        c = (a * fb - b * fa) / (fb - fa)
        # print(c)
        estimates.append(c)
        fc = f(c)
        if sign(fc) == sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc
        if len(estimates) >= 2 and abs(estimates[-1] - estimates[-2]) <= delta_x:
            break
    return c, estimates


def shooting_o2(F, a, alpha, b, beta, r0, u0, u1, delta=10E-2):
    def r(u):
        '''
        Boundary residual, as in equation (1)
        '''
        # Estimate theta_u
        # Evaluate y and y' until x=b, using initial condition y(a)=alpha and y'(a)=u
        X, Y = runge_kutta_4_2(F, a, array([alpha, u]), b, 0.01)
        theta_u = Y[-1, 0]  # last row, first column (y)
        return theta_u - beta

    # Find u as a the zero of r
    u, _ = false_position(r, u0, u1, delta)
    # print("u is")
    # print(u)

    # Now use u to solve the initial value problem one more time
    X, Y = runge_kutta_4_2(F, a, array([alpha, u]), r0, 0.01)
    return X, Y
