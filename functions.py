from typing import List

def sum_non_neg_diag(X: List[List[int]]) -> int:
    isExist = False
    sum = 0
    n = len(X)
    for i in range(n):
        if i < len(X[i]) and X[i][i] >= 0:
            sum += X[i][i]
            isExist = True
    if not isExist:
        return -1
    return sum



def count_elements(iterable):
    elements = {}
    for element in iterable:
        if element in elements:
            elements[element] += 1
        else:
            elements[element] = 1
    return elements

def are_multisets_equal(x, y):
    x1 = count_elements(x)
    y1 = count_elements(y)
    
    return x1 == y1



def max_prod_mod_3(x: List[int]) -> int:
    max_prod = float('-inf')  
    max_prod_mod3 = -1  

    for i in range(len(x) - 1):
        prod = x[i] * x[i+1]
        if x[i] % 3 == 0 or x[i+1] % 3 == 0:
            max_prod_mod3 = max(max_prod_mod3, prod)
        max_prod = max(max_prod, prod)

    if max_prod_mod3 != -1:
        return max_prod_mod3
    else:
        return -1



def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    return [[sum(image[i][j][k] * weights[k] for k in range(len(weights))) for j in range(len(image[i]))] for i in range(len(image))]



def rle_scalar(x: List[List[int]], y: List[List[int]]) -> int:
    x_vector = [val for val, count in x for _ in range(count)]
    y_vector = [val for val, count in y for _ in range(count)]

    if len(x_vector) != len(y_vector):
        return -1

    answer = sum(x * y for x, y in zip(x_vector, y_vector))
    return answer

def scalar_product(x: List[int], y: List[int]) -> int:
    answer = sum(x_i * y_i for x_i, y_i in zip(x, y))
    return answer

def vector_len(v: List[int]) -> float:
    answer = sum(x * x for x in v)
    return answer ** 0.5



def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    def dot_product(a, b):
        return sum(x * y for x, y in zip(a, b))

    def vector_norm(a):
        return (sum(x * x for x in a)) ** 0.5

    def cosine_similarity(a, b):
        if vector_norm(a) == 0 or vector_norm(b) == 0:
            return 1.0
        return dot_product(a, b) / (vector_norm(a) * vector_norm(b))

    result = []
    for x in X:
        row = []
        for y in Y:
            row.append(cosine_similarity(x, y))
        result.append(row)

    return result
