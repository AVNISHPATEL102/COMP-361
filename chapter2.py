# -*- coding: utf-8 -*-
"""Chapter2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/tgteacher/numerical-methods/blob/master/notebooks/Chapter2.ipynb

# Chapter 2: Systems of Linear Algebraic Equations

*Warning:* This is a critical chapter of the course.

## 2.1 Introduction

### Notation

A system of linear algebraic equations is written as follows:

$$
A_{11}x_1 + A_{12}x_2 + \ldots + A_{1n}x_n = b_1\\
A_{21}x_1 + A_{22}x_2 + \ldots + A_{2n}x_n = b_2\\
.\\
.\\
.\\
A_{n1}x_1 + A_{n2}x_2 + \ldots + A_{nn}x_n = b_n
$$

where $x_i$ are the unknowns.

It can be represented as $\textbf{Ax}=\textbf{b}$, where A is an $n \times n$ matrix, $\textbf{x}$ is a vector of n unknowns and $\textbf{b}$ is a vector of $n$ constants:

$$
    \begin{bmatrix}
        A_{11} & A_{12} & ... & A_{1n} \\
        A_{21} & A_{22} & ... & A_{2n} \\
        ...    & ...    & ... & ...    \\
        A_{n1} & A_{n2} & ... & A_{nn}
    \end{bmatrix}
    \begin{bmatrix}
    x_1\\
    x_2\\
    ...\\
    x_n
    \end{bmatrix}
    =
    \begin{bmatrix}
    b_1\\
    b_2\\
    ...\\
    b_n
    \end{bmatrix}
$$

It is also sometimes written using the *augmented form* combining $\textbf{A}$ and $\textbf{b}$ in a single matrix:
$$
\left[
\begin{array}{cccc|c}
 A_{11} & A_{12} & ... & A_{1n} & b_{1}\\
        A_{21} & A_{22} & ... & A_{2n} & b_{2} \\
        ...    & ...    & ... & ...    & ... \\
        A_{n1} & A_{n2} & ... & A_{nn} & b_{n}
\end{array}
\right]
$$

Let's do a brief [recap on linear algebra](Matrix-algebra.ipynb)...

### Existence and Uniqueness of Solution

A system of $n$ linear algebraic equations has a unique solution if and only if $\textbf{A}$ is *nonsingular*, that is: det($\textbf{A}$) $\neq$ 0.

If $\textbf{A}$ is singular, the system has no solution or an infinite number of solutions.

#### Exercice 2.1.2 (in class)

* Find examples of 2x2 matrices with a zero determinant.
* Find examples of systems with (1) no solution, (2) an infinite number of solutions.

[A few more words on linear algebra](Matrix-algebra.ipynb)...

### Conditioning

When the determinant of $\textbf{A}$ is "very small", small changes in the matrix result in large changes in the solution. In this case, the solution $\underline{\mathrm{cannot\ be\ trusted}}$.

The *condition number* of a matrix is defined as:

$$
\mathrm{cond}(\textbf{A}) = ||\textbf{A}||.||\textbf{A}^{-1}||
$$

If the condition number is close to unity, the matrix is well conditioned. On the contrary, a matrix with a large condition number is said to be $\underline{\mathrm{ill-conditioned}}$.

Let's write a function to compute the condition number of a matrix:
"""

def norm(a): # infinity norm
    n = 0
    for row in a:
        s = sum(abs(row))
        if s > n:
            n = s
    return n
                
def condition(a):
    from numpy.linalg import inv # this is cheating :), we'll see later how to compute the inverse of a matrix
    return norm(a)*norm(inv(a))

"""And let's compute the condition number of the following matrix:

$$
\textbf{A}=
\begin{bmatrix}
        1 & -1.001 \\
        2.001 & -2  
    \end{bmatrix}
$$

Using our function:
"""

from numpy import array
a = array([[1, -1.001], [2.001, -2]])
condition(a)

"""And using numpy:"""

from numpy import inf
from numpy.linalg import cond
cond(a, p=inf)

"""In practice, computing $\textbf{A}^{-1}$ is expensive. Conditioning is often gauged by comparing the determinant with the magnitude of the elements in the matrix: if the determinant is small compared to the elements of the matrix, the matrix will be gauged *ill-conditioned*.

#### Exercice: Effect of ill-conditioning on solutions.

1. Solve the linear system defined by $\textbf{Ax}$=$\textbf{b}$, where:
$
\textbf{A}=
\begin{bmatrix}
        1 & -1.001 \\
        2.001 & -2  
    \end{bmatrix}
$
and
$
\textbf{b}=
\begin{bmatrix}
        3 \\
        7  
    \end{bmatrix}
$
"""

from numpy import array
from numpy.linalg import solve # let's cheat for now, we'll program this in the next section

a = array([[1, -1.001], [2.001, -2]])
b = array([3, 7])
solve(a, b)

"""2. Solve the linear system defined by $\textbf{Ax}$=$\textbf{b}$, where:
$
\textbf{A}=
\begin{bmatrix}
        1 & -1.002 \\
        2.002 & -2  
    \end{bmatrix}
$
and
$
\textbf{b}=
\begin{bmatrix}
        3 \\
        7  
    \end{bmatrix}
$
"""

from numpy.linalg import solve # let's cheat for now, we'll program this in the next section

a = array([[1, -1.002], [2.002, -2]])
solve(a, b)

"""A 0.001 increment on matrix coefficients divided the solution by a factor of 2!

### Methods of Solution

There are two classes of methods to solve linear systems:
* Direct methods
* Iterative methods

Direct methods work by applying the following three operations to rewrite the system in a form that permits resolution:
* Exchanging two equations
* Multiplying an equation by a nonzero constant
* Subtracting an equation from another one

Iterative methods start with an initial solution $\textbf{x}$ and refine it until convergence. Iterative methods are generally used when the matrix is very large and sparse, for instance to solve the [PageRank](https://en.wikipedia.org/wiki/PageRank) equations.

#### Overview of Direct Methods

Direct methods are summarized in the Table below:

| Method        | Initial form    | Final form  |
| ------------- |:-------------:| -----:|
| Gauss Elimination      | $\textbf{Ax}=\textbf{b}$ | $\textbf{Ux}=\textbf{c}$ |
| LU decomposition      | $\textbf{Ax}=\textbf{b}$      |   $\textbf{LUx}=\textbf{b}$ |
| Gauss-Jordan Elimination | $\textbf{Ax}=\textbf{b}$      |    $\textbf{Ix}=\textbf{c}$ |

In this table, $\textbf{U}$ represents an upper triangular matrix, $\textbf{L}$ is a lower triangular matrix, and $\textbf{I}$ is the identity matrix. 

A square matrix is called triangular if it contains only zero elements below (lower triangular) or above (upper triangular) the diagonal. For instance, the following matrix is upper triangular:
$$
\textbf{U}=\begin{bmatrix}
1 & 2 & 3 \\
0 & 4 & 5 \\
0 & 0 & 6
\end{bmatrix}
$$
and the following matrix is lower triangular:
$$
\textbf{L}=\begin{bmatrix}
1 & 0 & 0 \\
2 & 3 & 0 \\
4 & 5 & 6
\end{bmatrix}
$$

Systems of the form $\textbf{Lx}$=$\textbf{c}$ can easily be solved by a procedure called $\underline{\mathrm{forward\ substitution}}$: the first equation has only a single unknown, which is easy to solve; after solving the first equation, the second one has only one unknown remaining, and so on.

#### Exercice

Solve the system of linear equations defined by $\textbf{Lx}$=$\textbf{c}$, where:
$
\textbf{L}=
\begin{bmatrix}
        1 & 0 & 0 \\
        2 & 4 & 0 \\
        3 & 1  & 1
    \end{bmatrix}
$ 
and
$
\textbf{b}=
\begin{bmatrix}
        1\\
        3\\
        2
    \end{bmatrix}
$
"""

# You can verify your solution as follows, but the point is to do it manually to understand
# how useful triangular matrices are!
from numpy.linalg import solve

a = array([[1, 0, 0], [2, 4, 0], [3, 1, 1]])
b = array([1, 3, 2])
solve(a, b)

"""Likewise, systems of the form $\textbf{Ux}$=$\textbf{b}$ can easily be solved by $\underline{\mathrm{back\  substitution}}$, solving the last equation first. 

For instance, solve $\textbf{Ux}=b$, where:
$
\textbf{U}=
\begin{bmatrix}
        3 & 1 & 1 \\
        0 & 4 & 2 \\
        0 & 0  & 1
    \end{bmatrix}
$ 
and
$
\textbf{b}=
\begin{bmatrix}
        1\\
        3\\
        2
    \end{bmatrix}
$
"""

U = array([[3, 1, 1], [0, 4, 2], [0, 0, 1]])
b = array([1, 3, 2])
solve(U, b)

"""Finally, systems of the form $\textbf{LUx}$=$\textbf{b}$ can quickly be solved by solving first $\textbf{Ly}$=$\textbf{b}$ by forward substitution, and then $\textbf{Ux}$=$\textbf{y}$ by back substitution.

### Exercice (Example 2.2 in textbook)

Solve the equations $\textbf{Ax}$=$\textbf{b}$, where:
$$
\textbf{A}=
\begin{bmatrix}
        8 & -6 & 2 \\
        -4 & 11 & -7 \\
        4 & -7  & 6
    \end{bmatrix}
    \quad
\textbf{b}=
\begin{bmatrix}
        28\\
        -40\\
        33
    \end{bmatrix}
$$ 
knowing that the LU decomposition of $\textbf{A}$ is (you should verify this):
$$
\textbf{A}=\textbf{LU}=
\begin{bmatrix}
        2 & 0 & 0 \\
        -1 & 2 & 0 \\
        1 & -1  & 1
    \end{bmatrix}
    \begin{bmatrix}
        4 & -3 & 1 \\
        0 & 4 & -3 \\
        0 & 0  & 2
    \end{bmatrix}
$$
(done manually in class)
"""

# You should be able to do this manually
from numpy import array, dot
a = array([[8, -6, 2], [-4, 11, -7], [4, -7, 6]])
b = array([28, -40, 33])
l = array([[2, 0, 0], [-1, 2, 0], [1, -1, 1]])
u = array([[4, -3, 1], [0, 4, -3], [0, 0, 2]])
# Check if the factorization is correct
dot(l, u)

# Solving without the LU decomposition
from numpy.linalg import solve
solve(a, b)

# Solving with the decomposition
y = solve(l, b)
solve(u, y)

"""## 2.2 Gauss Elimination

This method consists of two phases:
1. The elimination phase, to transform the equations in the form $\textbf{Ux}$=$\textbf{c}$,
2. The back substitution phase, to solve the equations.

### Elimination phase

The elimination phase multiplies equations by a constant and subtracts them, which is represented as follows:
$$
Eq.(i) \leftarrow Eq.(i) - \lambda Eq.(j)
$$
Equation $\textit{j}$ is called the $\textit{pivot}$.

For instance, let's consider the following equations:

$$
  3x_1+2x_2-7x_3=4  \quad (a)\\
  2x_1-x_2-4x_3=1   \quad (b)\\
  -x_1-3x_2+x_3=3   \quad (c)
$$

We start the process by choosing Equation (a) as the pivot, and chosing $\lambda$ to eliminate $x_1$ from Equations (b) and (c):
$$
Eq.(b) \leftarrow Eq.(b) - \frac{2}{3}Eq.(a)\\
Eq.(c) \leftarrow Eq.(c) - \left( -\frac{1}{3} \right)Eq.(a)
$$

which gives:
$$
  3x_1+2x_2-7x_3=4  \quad (a)\\
  -\frac{7}{3}x_2+\frac{2}{3}x_3=-\frac{5}{3}   \quad (b)\\
  -\frac{7}{3}x_2-\frac{4}{3}x_3=\frac{13}{3}   \quad (c)
$$

We now reiterate the process by choosing Equation (b) as the pivot to eliminate $x_2$ from Equation (c):
$$
Eq.(c) \leftarrow Eq.(c) - Eq.(b)
$$

which gives:
$$
  3x_1+2x_2-7x_3=4  \quad (a)\\
  -\frac{7}{3}x_2+\frac{2}{3}x_3=-\frac{5}{3}   \quad (b)\\
  -2x_3=\frac{18}{3}   \quad (c)
$$
The elimination phase is complete.

Using the augmented notation, the process is written as follows:
$$
\left[
\begin{array}{ccc|c}
3 & 2 & -7 & 4 \\
2 & -1 & -4 & 1 \\
-1 & -3 & 1 & 3
\end{array}
\right]
$$

$$
\left[
\begin{array}{ccc|c}
3 & 2 & -7 & 4 \\
0 & -7/3 & 2/3 & -5/3 \\
0 & -7/3 & -4/3 & 13/3
\end{array}
\right]
$$

$$
\left[
\begin{array}{ccc|c}
3 & 2 & -7 & 4 \\
0 & -7/3 & 2/3 & -5/3 \\
0 & 0 & -2 & 18/3
\end{array}
\right]
$$

### Back substitution

We can now solve the equations by back substitution:
$$
x_3 = -\frac{9}{3} = -3 \quad (c) \\
x_2 = -\frac{3}{7} \left(-\frac{5}{3} + 2 \right) = - \frac{1}{7}\quad (b) \\
x_1 = \frac{1}{3}\left( 4 + \frac{2}{7} - 21\right) = -\frac{39}{7} \quad (a)
$$

### Note

* The elimination phase leaves the determinant of the matrix unchanged.
* The determinant of a triangular matrix is the product of its diagonal coefficients.

Thus:

$$
\mathrm{det}\left(
\begin{bmatrix}
3 & 2 & -7 \\
2 & -1 & -4 \\
-1 & -3 & 1
\end{bmatrix}
\right) = 3.-\frac{7}{3}.-2 = 14
$$
"""

from numpy.linalg import det
from numpy import array
a = array([[3, 2, -7], [2, -1, -4], [-1, -3, 1]])
det(a)

"""### Algorithm for Gauss Elimination Method

#### Elimination Phase

Let's look at the equations at iteration $k$ of the elimination process.

* The first $k$ rows have already been transformed.
* Row $k$ is the pivot row
* Row $i$ is the row being transformed

The augmented coefficient matrix is:
$$
\left[
\begin{array}{ccccccccc|c}
A_{11} & A_{12} & A_{13} & \ldots & A_{1k} & \ldots & A_{1j} & \ldots & A_{1n} & b_1 \\
0      & A_{22} & A_{23} & \ldots & A_{2k} & \ldots & A_{2j} & \ldots & A_{2n} & b_2 \\
0      & 0      & A_{33} & \ldots & A_{3k} & \ldots & A_{3j} & \ldots & A_{3n} & b_3 \\
\ldots \\
0      & 0      & 0 & \ldots & A_{kk} & \ldots & A_{kj} & \ldots & A_{kn} & b_k \\
\hline
\ldots \\
0      & 0      & 0 & \ldots & A_{ik} & \ldots & A_{ij} & \ldots & A_{in} & b_i \\
\ldots \\
0      & 0      & 0 & \ldots & A_{nk} & \ldots & A_{nj} & \ldots & A_{nn} & b_n
\end{array}
\right]
$$
Note: the coefficients are not the one of the original matrix.

##### Principle
* Use the first $n-1$ rows successively as pivot row $k$.
* For each pivot row $k$, transform row $i$ as follows ($k+1\leq i \leq n$): 
$$
A_{ij} \leftarrow A_{ij} - \lambda_{ik} A_{kj}, \forall j \in |[k, n]|
$$
* With $\lambda_{ik}$ such that after transformation, $A_{ik}=0$: 
$$
\lambda_{ik} = \frac{A_{ik}}{A_{kk}}
$$.

##### Implementation
"""

from numpy import shape
def gauss_elimination(a, b, verbose=False):
    n, m = shape(a)
    n2,  = shape(b)
    assert(n==n2)
    for k in range(n-1):
        for i in range(k+1, n):
            assert(a[k,k] != 0) # woops, what happens in this case? we'll talk about it later!
            if (a[i,k] != 0): # no need to do anything when lambda is 0
                lmbda = a[i,k]/a[k,k] # lambda is a reserved keyword in Python
                a[i, k:n] = a[i, k:n] - lmbda*a[k, k:n] # list slice operations
                b[i] = b[i] - lmbda*b[k] # don't forget this step! 
            if verbose:
                print(a, b)

"""##### Example"""

a = array([[3.0, 2, -7], [2, -1, -4], [-1, -3, 1]])
b = [4, 1, 3]
gauss_elimination(a, b)
print(a,b)

"""#### Back substition phase

After the elimination phase, the augmented matrix has the following form:
$$
\left[
\begin{array}{ccccc|c}
A_{11} & A_{12} & A_{13} & \ldots & A_{1n} & b_1 \\
0      & A_{22} & A_{23} & \ldots & A_{2n} & b_2 \\
0      & 0      & A_{33} & \ldots & A_{3n} & b_3 \\
\ldots\\
0      & 0      & 0 & \ldots & A_{nn} & b_n
\end{array}
\right]
$$

The equations are solved from the last row to the first:
* $x_n$ = $b_n/A_{nn}$
* $\forall i < n, \quad x_i = \left( b_i - \sum_{j = i +1}^n{A_{ij}x_{j}} \right) \frac{1}{A_{ii}}$

##### Implementation
"""

from numpy import dot, zeros, shape
def gauss_substitution(a, b):
    n, m = shape(a)
    n2, = shape(b)
    assert(n==n2)
    x = zeros(n)
    for i in range(n-1, -1, -1): # decreasing index
        x[i] = (b[i] - dot(a[i,i+1:], x[i+1:]))/a[i,i]
    return x

"""##### Exercice

Use Gauss elimination to solve $\textbf{Ax}=\textbf{b}$ where:
$$
\textbf{A} = \begin{bmatrix}
6 & -4 & 1\\
-4 & 6 & -4 \\
1 & -4 & 6
\end{bmatrix}
\quad
\mathrm{and}
\quad
\textbf{b} = \begin{bmatrix}
-14 \\
36\\
6
\end{bmatrix}
$$
(done manually in class)
"""

a = array([[6.0, -4, 1], [-4, 6, -4], [1, -4, 6]])
b = [-14, 36, 6]
gauss_elimination(a, b)
gauss_substitution(a, b)

# let's test our implementation
from numpy.linalg import solve
solve(a, b)

"""#### Complete Solver"""

def gauss(a, b):
    gauss_elimination(a, b)
    return gauss_substitution(a, b)

a = array([[6.0, -4, 1], [-4, 6, -4], [1, -4, 6]])
b = [-14, 36, 6]
gauss(a, b)

"""### Operations count

Counting multiplications and divisions only:

Elimination phase:
$$
\sum_{k=1}^n \sum_{i=k+1}^n (n-k) + 2 = \sum_{k=1}^n (n-k)^2 + 2(n-k) = \sum_{k=0}^{n-1} k^2 + 2\sum_{k=0}^{n-1} k
=\frac{(n-1)n(2n-1)}{6}+n(n-1) = \frac{n(n-1)(2n+5)}{6} = O\left(\frac{n^3}{3}\right)
$$

Backsubstitution phase:
$\sum_{k=1}^n k=\frac{n(n+1)}{2}=O\left(\frac{n^2}{2}\right)$

### Multiple Sets of Equations

It is frequently necessary to solve $\textbf{Ax}=\textbf{b}$ for multiple values of $\textbf{b}$. We denote multiple sets of equations by 
$\textbf{AX}=\textbf{B}$, where:

$$
\textbf{X}=[\textbf{x}_1, \textbf{x}_2, \ldots \textbf{x}_m] \quad \textbf{B}=[\textbf{b}_1, \textbf{b}_2, \ldots \textbf{b}_m]
$$
are $n \times m$ matrices.

An easy way to handle such equations during the elimination phase is to include all constant vectors in the augmented matrix, so that they are transformed simultaneously with the coefficient matrix:
$$
\left[
\begin{array}{ccccccccc|cccc}
A_{11} & A_{12} & A_{13} & \ldots & A_{1k} & \ldots & A_{1j} & \ldots & A_{1n} & b_{11} & b_{12} & \ldots & b_{1m} \\
0      & A_{22} & A_{23} & \ldots & A_{2k} & \ldots & A_{2j} & \ldots & A_{2n} & b_{21} & b_{22} & \ldots & b_{2m} \\
0      & 0      & A_{33} & \ldots & A_{3k} & \ldots & A_{3j} & \ldots & A_{3n} & b_{31} & b_{32} & \ldots & b_{3m} \\
\ldots \\
0      & 0      & 0 & \ldots & A_{kk} & \ldots & A_{kj} & \ldots & A_{kn} & b_{k1} & b_{k2} & \ldots & b_{km} \\ \\
\hline
\ldots \\
0      & 0      & 0 & \ldots & A_{ik} & \ldots & A_{ij} & \ldots & A_{in} & b_{i1} & b_{i2} & \ldots & b_{im} \\ \\
\ldots \\
0      & 0      & 0 & \ldots & A_{nk} & \ldots & A_{nj} & \ldots & A_{nn} & b_{n1} & b_{n2} & \ldots & b_{nm} 
\end{array}
\right]
$$


$\underline{\mathrm{The\ LU\ decomposition\ method\ provides\ a\ more\ versatile\ solution}}$.

## 2.3 LU Decomposition Methods

### Introduction

Any square matrix $\textbf{A}$ can be expressed as a product of a lower triangular matrix $\textbf{L}$ and an upper triangular matrix $\textbf{U}$:
$$
\textbf{A}=\textbf{LU}
$$

However, this decomposition is not unique unless certain constraints are placed on $\textbf{L}$ and $\textbf{U}$. Common decomposition methods are listed here:

| Name        | Constraints    |
| ------------- |:-------------:|
| Doolittle's decomposition  | $\textbf{L}_{ii}=1$ |
| Crout's decomposition      | $\textbf{U}_{ii}=1$      |
| Choleski decomposition | $\textbf{L}=\textbf{U}^T$ |

From the decomposition of $\textbf{A}$, it is easy to solve $\textbf{A}\textbf{x}=\textbf{b}$:
1. Solve $\textbf{Ly}=\textbf{b}$ by forward substitution
2. Solve $\textbf{Ux}=\textbf{y}$ by back substitution

The advantage of the $\textbf{LU}$ decomposition method is that once $\textbf{A}$ is decomposed, the cost of solving the system for another instance of $\textbf{b}$ is relatively small.

### Doolittle's Decomposition Method

#### Decomposition phase

This method is closely related to Gauss elimination. Let's consider a $3 \times 3$ matrix $\textbf{A}$ and let's assume that there exists $\textbf{L}$ and $\textbf{U}$:
$$
\textbf{L} = \begin{bmatrix}
1 & 0 & 0 \\
L_{21} & 1 & 0 \\
L_{31} & L_{32} & 1
\end{bmatrix}
\quad
\textbf{U} = \begin{bmatrix}
U_{11} & U_{12} & U_{13}\\
0      & U_{22} & U_{23}\\
0      & 0      & U_{33}
\end{bmatrix}
$$
such that $\textbf{A}=\textbf{LU}$. We get:
$$
\textbf{A} = \begin{bmatrix}
U_{11} & U_{12} & U_{13} \\
L_{21}U_{11} & L_{21}U_{12}+U_{22} & L_{21}U_{13}+U_{23}\\
L_{31}U_{11} & L_{31}U_{12}+L_{32}U_{22} & L_{31}U_{13}+L_{32}U_{23}+U_{33}
\end{bmatrix}
$$

Let's now apply Gauss elimination.

First pass:
* row2 $\leftarrow$ row2 - $L_{21}$row1
* row3 $\leftarrow$ row3 - $L_{31}$row1

$$
\textbf{A} = \begin{bmatrix}
U_{11} & U_{12} & U_{13} \\
0 & U_{22} & U_{23}\\
0 & L_{32}U_{22} & L_{32}U_{23}+U_{33}
\end{bmatrix}$$

Second pass:
* row3 $\leftarrow$ row3-$L_{32}$row2

$$
\textbf{A} = \begin{bmatrix}
U_{11} & U_{12} & U_{13} \\
0 & U_{22} & U_{23}\\
0 & 0 & U_{33}
\end{bmatrix}$$

This illustration shows 2 properties of Doolittle's decomposition:
1. $\textbf{U}$ is the upper diagonal matrix resulting from Gauss decomposition
2. The off-diagonal elements of $\textbf{L}$ are the multipliers for the pivot equations: $L_{ij}$ is used to eliminate $A_{ij}$

It is common to store the multipliers in the lower triangular portion of the matrix. The final form of the coefficient matrix would be:
$$
[\textbf{L} \backslash \textbf{U}]= \begin{bmatrix}
U_{11} & U_{12} & U_{13} \\
L_{21} & U_{22} & U_{23} \\
L_{31} & L_{32} & U_{33}
\end{bmatrix}
$$
The algorithm is very similar to Gauss elimination:
"""

from numpy import shape
def dolittle_decomp(a, verbose=False):
    n, m = shape(a)
    assert(n == m) # check that matrix is square
    for k in range(n-1):
        for i in range(k+1, n):
            assert(a[k,k] != 0) # woops, what happens in this case? we'll talk about it later!
            if (a[i,k] != 0): # no need to do anything when lambda is 0
                lmbda = a[i,k]/a[k,k] # lambda is a reserved keyword in Python
                a[i, k:n] = a[i, k:n] - lmbda*a[k, k:n] # list slice operations
                a[i, k] = lmbda # <--- new in Dolittle decomposition
            if verbose:
                print(a)
    return a

from numpy import array
a = array([[1.0, 4, 1], [1, 6, -1], [2, -1, 2]])
dolittle_decomp(a)
print(a)

"""#### Solution phase

First we solve $\textbf{Ly}=\textbf{b}$ by forward substitution:
$$
y_1 = b_1 \\
L_{21}y_1 + L_{22}y_2 = b_2 \\
\ldots
L_{k1}y_1 + L_{k2}y_2 + \ldots + L_{k,k-1}y_{k-1} + y_k = b_k \\
\ldots
$$

Thus:
$$
y_k = b_k - \sum_{i=1}^{k-1}L_{k,i}y_i
$$

Implementation:
"""

def dolittle_solution_forward(l, b):
    for k in range(1, len(b)):
        b[k] = b[k] - dot(l[k,0:k],b[0:k])
    return b

"""The back substitution phase for solving $\textbf{Ux}=\textbf{y}$ is identical to the one in the Gauss elimination method.

#### Complete solver
"""

def dolittle_solver(a, b, verbose=False):
    dolittle_decomp(a)
    if verbose:
        print("LU={}".format(a))
    y = dolittle_solution_forward(a, b)
    if verbose:
        print("y={}".format(y))
    return gauss_substitution(a, y)

"""#### Exercise

Use Dolittle's decomposition to solve $\textbf{Ax}=\textbf{b}$ where:
$$
\textbf{A} = \begin{bmatrix}
1 & 4 & 1 \\
1 & 6 & -1 \\
2 & -1 & 2
\end{bmatrix}
\quad
\textbf{b} = \begin{bmatrix}
7 \\
13 \\
5
\end{bmatrix}
$$
"""

from numpy import array
a = array([[1.0, 4, 1], [1, 6, -1], [2, -1, 2]])
b = array([7, 13, 5])
dolittle_solver(a, b, True)

# and let's test our implementation
from numpy.linalg import solve
from numpy import array
a = array([[1.0, 4, 1], [1, 6, -1], [2, -1, 2]])
b = array([7, 13, 5])
solve(a, b)

"""### Choleski's Decomposition Method

Choleski's LU decomposition focuses on $\underline{\mathrm{symmetric,\ definite,\ positive}}$ matrices. It does $O\left( \frac{n^3}{6}\right)$ multiplications or divisions. Compared to Gauss elimination, it reduces the number of operations by about half.

The decomposition is in the following form:
$$
\textbf{A}=\textbf{L}\textbf{L}^T
$$
which gives, for a $3 \times 3$ matrix:
$$
\begin{bmatrix}
A_{11} & A_{12} & A_{13} \\
A_{12} & A_{22} & A_{23} \\
A_{13} & A_{23} & A_{33}
\end{bmatrix}
= \begin{bmatrix}
L_{11} & 0 & 0 \\
L_{12} & L_{22} & 0  \\
L_{13} & L_{23} & L_{33}
\end{bmatrix}
\begin{bmatrix}
L_{11} & L_{12} & L_{13} \\
0      & L_{22} & L_{23} \\
0      & 0 & L_{33}
\end{bmatrix}
$$

Thus:
$$
\begin{bmatrix}
A_{11} & A_{12} & A_{13} \\
A_{12} & A_{22} & A_{23} \\
A_{13} & A_{23} & A_{33}
\end{bmatrix}
= \begin{bmatrix}
L_{11}^2 & L_{11}L_{12} & L_{11}L_{13} \\
L_{12}L_{11}  & L_{12}^2 + L_{22}^2 & L_{12}L_{13} + L_{22}L_{23} \\
L_{13}L_{11} & L_{13}L_{12} + L_{23}L_{22} & L_{13}^2 + L_{23}^2 + L_{33}^2
\end{bmatrix}
$$

And by identification in the first column:

$$
A_{11} = L_{11}^2 \\
A_{12} = L_{11}L_{12}\\
A_{13} = L_{11}L_{13}
$$
Thus:
$$
L_{11} = \sqrt{A_{11}} \\
L_{12} = \frac{A_{12}}{L_{11}} \\
L_{13} = \frac{A_{13}}{L_{11}}
$$

By identification of the second column:
$$
A_{22} = L_{12}^2 + L_{22}^2 \\
A_{23} = L_{13}L_{12} + L_{23}L_{22}
$$
Thus:
$$
L_{22} = \sqrt{A_{22} - L_{12}^2}\\
L_{23} = \frac{A_{23} - L_{13}L_{12}}{L_{22}}
$$

By identification of the last column:
$$
A_{33} = L_{13}^2 + L_{23}^2 + L_{33}^2
$$
Thus:
$$
L_{33} = \sqrt{A_{33}-L_{13}^2-L_{23}^2}
$$

Likewise, for matrices of size $n \times n$, the solution is:
$$
L_{ii} = \sqrt{A_{ii}-\sum_{j=1}^{i-1}{L_{ji}^2}}, \quad \forall i \in |[2, n]|
$$
with:
$$
L_{11} = \sqrt{A_{11}}
$$

And for a non-diagonal term:

$$
L_{ij} = \frac{A_{ij}-\sum_{k=1}^{j-1}{L_{ik}L_{jk}}}{L_{jj}},\quad \forall j \in |[2, n-1|] \quad \mathrm{and} \quad \forall i \in |[j+1, n]|
$$

#### Example (2.6 in the Textbook)

Compute the Cholesky decomposition of:
$$
A = \begin{bmatrix}
4 & -2 & 2 \\
-2 & 2 & -4 \\
2 & -4 & 11
\end{bmatrix}
$$
"""

# You should be able to do the decomposition on paper (shown in class)
# with numpy:
from numpy.linalg import cholesky
a = array([[4, -2, 2], [-2, 2, -4], [2, -4, 11]])
cholesky(a)

"""## 2.4 Symmetric and Banded Coefficient Matrices

A matrix is said to be $\underline{\mathrm{banded}}$ when all its non-zero coefficients are clustered around its diagonal, for instance (X are the non-zero elements):
$$
\textbf{A} = \begin{bmatrix}
X & X & 0 & 0 & 0 \\
X & X & X & 0 & 0 \\
0 & X & X & X & 0 \\
0 & 0 & X & X & X \\
0 & 0 & 0 & X & X
\end{bmatrix}
$$

If a banded matrix is decomposed in the form $\textbf{A}=\textbf{LU}$ then $\textbf{L}$ and $\textbf{U}$ preserve the banded structure. For instance, the matrix $\textbf{A}$ above would be decomposed as follows:
$$
\textbf{L} = \begin{bmatrix}
X & 0 & 0 & 0 & 0 \\
X & X & 0 & 0 & 0 \\
0 & X & X & 0 & 0 \\
0 & 0 & X & X & 0 \\
0 & 0 & 0 & X & X
\end{bmatrix}
\quad
\textbf{U} = \begin{bmatrix}
X & X & 0 & 0 & 0 \\
0 & X & X & 0 & 0 \\
0 & 0 & X & X & 0 \\
0 & 0 & 0 & X & X \\
0 & 0 & 0 & 0 & X
\end{bmatrix}
$$

### Tridiagnoal Coefficient Matrix

Memory can be saved by storing only the non-zero elements of banded matrices.

Consider the $n \times n$ tridiagonal matrix $\textbf{A}$ noted as follows:
$$
\textbf{A} = \begin{bmatrix}
d_1 & e_1 & 0 & 0 & \ldots & 0 \\
c_1 & d_2 & e_2 & 0 & \ldots & 0 \\
0 & c_2 & d_3 & e_3 & \ldots & 0 \\
0 & 0 & c_3 & d_4 & \ldots & 0 \\
\ldots \\
0 & 0 & \ldots & 0 & c_{n-1} & d_n
\end{bmatrix}
$$

The elements of $\textbf{A}$ are stored in three vectors $\textbf{a}$, $\textbf{b}$ and $\textbf{c}$, which significantly saves on memory.

Let us now apply Dolittle's LU decomposition. We eliminate $c_{k-1}$ from row $k$ as follows:
$$
row\ k \leftarrow row\ k - (c_{k-1}/d_{k-1}) \times row(k-1), \quad \forall k \in |[2, n]|
$$

$e_k$ is not affected while $d_k$ is modified as follows:
$$
d_k \leftarrow d_k - (c_{k-1}/d_{k-1})e_{k-1}
$$

To finish Doolitle's decomposition of the form [$\textbf{L}\backslash\textbf{U}$], we store the multiplier 
$\lambda_k=c_{k-1}/d_{k-1}$ in the location of $c_{k-1}$:
$$
c_{k-1} \leftarrow c_{k-1}/d_{k-1}
$$

The decomposition algorithm is implemented as follows:
"""

def tridiag_decomp(c, d, e):
    assert(len(c) == len(d) == len(e))
    n = len(c)
    for k in range(1, n):
        lambd = c[k-1]/d[k-1]
        d[k] -= lambd*e[k-1]
        c[k-1] = lambd

"""Decomposition phase is now in O(n)!

### Solution phase

We start by solving $\textbf{Ly}=\textbf{b}$:

$$
\left[ \textbf{L} | \textbf{b} \right] = \left[ \begin{array}{cccccc|c}
1 & 0 & 0 & 0 & \ldots & 0 & b_1 \\
c_1 & 1 & 0 & 0 & \ldots & 0 & b_2 \\
0 & c_2 & 1 & 0 & \ldots & 0 & b_3 \\
0 & 0 & c_3 & 1 & \ldots & 0 & b_4 \\
\ldots \\
0 & 0 & \ldots & 0 & c_{n-1} & 1 & b_n
\end{array}
\right]
$$
Note that $c_i$ represents the multipliers computed during the decomposition, not the original coefficients in $\textbf{A}$.

We have:
$$
y_1 = b_1\\
\forall i \in |[2, n|], \quad y_i = b_i - c_{i-1}y_{i-1}
$$

We now solve $\textbf{Ux}=\textbf{y}$:

$$
\left[ \textbf{U} | \textbf{y} \right] = \left[ \begin{array}{cccccc|c}
d_1 & e_1 & 0 & 0 & \ldots & 0 & y_1 \\
0 & d_2 & e_2 & 0 & \ldots & 0 & y_2 \\
0 & 0 & d_3 & e_3 & \ldots & 0 & y_3 \\
\ldots \\
0 & 0 & \ldots & 0 & d_{n-1} & e_{n-1} & y_{n-1}\\
0 & 0 & \ldots & 0 & 0 & d_{n} & y_n
\end{array}
\right]
$$

which gives, by backsubstitution:

$$
x_n = \frac{y_n}{d_n} \\
\forall i \in |[1, n-1]|, \quad x_i = \frac{y_{i} - e_ix_{i+1}}{d_i}
$$

### Implementation
"""

from numpy import zeros
def tridiag_solve(c, d, e, b): # watch out, input has to be in LU form!
    assert(len(c) == len(d) == len(e) == len(b))
    n = len(c)
    # forward substitution
    for i in range(1, n):
        b[i] = b[i]-c[i-1]*b[i-1] # Here we use b to store y
    # back substitution
    b[n-1] = b[n-1]/d[n-1] # Here we use b to store x
    for i in range (n-2, -1, -1):
        b[i] = (b[i]-e[i]*b[i+1])/d[i]

"""Solution phase is in O(n) too!"""

def tridiag_solver(c, d, e, b): # complete solver for tridiagonal systems
    tridiag_decomp(c, d, e)
    tridiag_solve(c, d, e, b)

import numpy as np
a = np.array([[1, 1, 0, 0, 0], [1, 2, 1, 0, 0], [0, 1, 3, 1, 0], [0, 0, 1, 4, 1], [0, 0, 0, 1, 5]])
c = np.array([1, 1, 1, 1, 0], dtype=np.float_)
d = np.array([1, 2, 3, 4, 5], dtype=np.float_)
e = np.array([1, 1, 1, 1, 0], dtype=np.float_)
b = np.array([1, 2, 3, 4, 5], dtype=np.float_)
tridiag_solver(c, d, e, b)
print("Our result:", b)

from numpy.linalg import solve
b = array([1, 2, 3, 4, 5], dtype=np.float_) # watch out, b is modified by our solver!
x = solve(a, b)
print("Numpy's result:", x)

"""## 2.5 Pivoting

Any idea what is going wong below? 

Let's define the following system:
"""

import numpy as np
a = array([[2, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=np.float_)
b = array([1, 0, 0], dtype=np.float_)
print(a)
print(b)

"""Let's solve it using our `gauss` function:"""

x = gauss(a, b)
print(x)

"""Let's check that it is the correct answer:"""

import numpy as np
a = array([[2, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=np.float_)
b = array([1, 0, 0], dtype=np.float_)
x = np.linalg.solve(a, b)
print(x)

"""Now let's swap  rows 1 and 3 of the system:"""

import numpy as np
a = array([[0, -1, 1], [-1, 2, -1],  [2, -1, 0]], dtype=np.float_)
b = array([0, 0, 1], dtype=np.float_)
print(a)
print(b)

"""The solution remains the same:"""

import numpy as np
x = np.linalg.solve(a, b)
print(x)

"""But with our implementation, woops!"""

x = gauss(a, b)
print(x)

"""Issue: we don't want zeros, or elements with a very small absolute value, on the diagonal (small elements would be detrimental to numerical accuracy).

To address this issue, we will swap rows in $\textbf{A}$: this is called $\underline{\mathrm{pivoting}}$.

### Diagonal dominance

An $n \times n$ matrix is $\underline{\mathrm{diagonal\ dominant}}$ if:
$$
\forall i \in |[1, n]|, \quad |A_{ii}| > \sum_{j \neq i}|A_{ij}|
$$

It can be shown that if $\textbf{A}$ is diagonal dominant, then the solution of $\textbf{Ax}=\textbf{b}$ does not benefit from row pivoting.

### Gauss Elimination with scale row pivoting

Let's define a vector $\textbf{s}$ such that:
$$
\forall i \in |[1, n]|, \quad s_i = \max_j{|A_{ij}|}
$$

And let's define the relative size of an element $A_{ij}$ as follows:
$$
r_{ij} = \frac{|A_{ij}|}{s_i}
$$

Before starting iteration $k-1$ of the Gauss elimination process, the linear system is defined as follows:
$$
\left[
\begin{array}{ccccccccc|c}
A_{11} & A_{12} & A_{13} & \ldots & A_{1k} & \ldots & A_{1j} & \ldots & A_{1n} & b_1 \\
0      & A_{22} & A_{23} & \ldots & A_{2k} & \ldots & A_{2j} & \ldots & A_{2n} & b_2 \\
0      & 0      & A_{33} & \ldots & A_{3k} & \ldots & A_{3j} & \ldots & A_{3n} & b_3 \\
\ldots \\
\hline\\
0      & 0      & 0 & \ldots & A_{kk} & \ldots & A_{kj} & \ldots & A_{kn} & b_k \\
\ldots \\
0      & 0      & 0 & \ldots & A_{ik} & \ldots & A_{ij} & \ldots & A_{in} & b_i \\
\ldots \\
0      & 0      & 0 & \ldots & A_{nk} & \ldots & A_{nj} & \ldots & A_{nn} & b_n
\end{array}
\right]
$$

Instead of choosing $A_{kk}$ as the pivot, we will choose the row with the largest $r_{ik}$, that is, we choose row $p$ such that:
$$
r_{pk} = \max_{i \geq k}(r_{ik})
$$
If we find such an element, we exchange rows $k$ and $p$ in $\textbf{A}$ and in $\textbf{s}$.

Let's write a function to swap rows of a matrix:
"""

from numpy import shape
def swap(a, i, j):
    if len(shape(a)) == 1:
        a[i],a[j] = a[j],a[i] # unpacking
    else:
        a[[i, j], :] = a[[j, i], :]

"""And let's use it to rewrite the Gauss elimination function:"""

from numpy import shape, argmax
def gauss_elimination_pivot(a, b, verbose=False):
    n, m = shape(a)
    n2,  = shape(b)
    assert(n==n2)
    # New in pivot version
    s = zeros(n)
    for i in range(n):
        s[i] = max(abs(a[i, :]))
    for k in range(n-1):
        # New in pivot version
        p = argmax(abs(a[k:,k])/s[k:]) + k
        swap(a, p, k)
        swap(b, p, k)
        swap(s, p, k)
        # The remainder remains as in the previous version
        for i in range(k+1, n):
            assert(a[k,k] != 0) # this shouldn't happen now, unless the matrix is singular
            if (a[i,k] != 0): # no need to do anything when lambda is 0
                lmbda = a[i,k]/a[k,k] # lambda is a reserved keyword in Python
                a[i, k:n] = a[i, k:n] - lmbda*a[k, k:n] # list slice operations
                b[i] = b[i] - lmbda*b[k]
            if verbose:
                print(a, b)

def gauss_pivot(a, b):
    gauss_elimination_pivot(a, b)
    return gauss_substitution(a, b) # as in the previous version

import numpy as np
a = array([[0, -1, 1], [-1, 2, -1],  [2, -1, 0]], dtype=np.float_)
b = array([0, 0, 1], dtype=np.float_)
try:
    x = gauss(a, b)
except AssertionError as e:
    print("Version without pivot crashes as expected")
a = array([[0, -1, 1], [-1, 2, -1],  [2, -1, 0]], dtype=np.float_)
b = array([0, 0, 1], dtype=np.float_)
x = gauss_pivot(a, b)
print("Solution with scaled pivoting: ", x)

"""#### When to pivot

Drawbacks of pivoting:
* Increased computation cost
* Destruction of symmetry and banded structure

There is no absolute rule to determine when pivoting should be used. Rules of thumb:
Don't pivot when matrix is:
* Banded
* Positive definite
* Symmetric

## 2.6 Matrix inversion

The inverse of an $n \times n$ matrix $\textbf{A}$ can be computed by solving the linear equations defined by:
$$
\textbf{AX}=\textbf{I}
$$
where $\textbf{I}$ is the $n \times n$ identity matrix. 

Warning:
* This is a compute-intensive task!
* If $\textbf{A}$ is banded, $\textbf{A}^{-1}$ is not banded in genral.
"""