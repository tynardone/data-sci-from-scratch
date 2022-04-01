from typing import List
import math

Vector = List[float] # type aliasing, Vector will be a list of floats

num_friends = [100.0,49,41,40,25,21,21,19,19,18,
18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,
10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,
9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
8,8,8,8,8,8,8,8,8,8,8,8,8,
7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
2,2,2,2,2,2,2,2,2,2,2,2
,2,2,2,2,2,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


heigh_weight_age = [120, 170, 40]
grades = [95, 80, 75, 62]

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements of two vectors"""
    assert len(v) == len(w), "vectors are required to be same length"
    return[v_i + w_i for v_i, w_i in zip(v,w)]

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts the corresponding elements"""
    assert len(v) == len(w), "vectors are required to be the same length"
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all the corresponding elements"""
    # check that vectors is not empty
    assert vectors, "no vectors provided"
    
    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v)==num_elements for v in vectors), "different size vectors not allowed"
    
    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by scalar c"""
    return [c * v_i for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

# unit tests for vector functions defined above
assert add([1,2,3],[4,5,6]) == [5,7,9]
assert subtract([5,7,9], [4,5,6]) == [1,2,3]
assert vector_sum([[1, 2,], [3,4], [5,6], [7,8]]) == [16, 20]
assert scalar_multiply(2, [1,2,3]) == [2,4,6]
assert vector_mean([[1,2], [3,4], [5,6]]) == [3,4]

    

def dot(v: Vector, w: Vector) -> float:
    """Computes sum of all component-wise multiplications"""
    assert len(v) == len(w)
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v: Vector) -> float:
    """Computes the sum of every element squares"""
    return dot(v, v)

# unit tests for vector functions defined above
assert dot([1, 2], [3, 4]) == 11
assert sum_of_squares([1, 2, 3]) == 14

def magnitude(v: Vector) -> float:
    """Returns the magnitude of v"""
    return math.sqrt(sum_of_squares(v))

def squared_distance(v: Vector, w: Vector) -> float:
    """Computes the sum of squared element-wise subtraction"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v,w))


assert magnitude([3,4]) == 5 

# Matrices
# Two-dimensional collection of numbers. 
# Will be represented as a list of lists, each inner list the 
# same length and representing a row. If A is a matrix A[ i ][ j ] is 
# the element in the ith row and jth column.

# Another type alias
from typing import Tuple, Callable

Matrix = List[List[float]]

A = [[1,2,3],
     [4,5,6]]

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0])
    return num_rows, num_cols

def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A as a Vector"""
    return A[i]

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A as a Vector"""
    return [A_i[j] for A_i in A ]

def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    """Return a num_rows by num_cols matrix whose i,jth entry is entry_fn(i,j)"""
    return [[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)]

def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i==j else 0)

assert shape([[1,2,3],[4,5,6]]) == (2, 3)
assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1]]



