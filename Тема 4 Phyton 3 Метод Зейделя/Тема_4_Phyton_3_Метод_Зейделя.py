
import numpy as np

def seidel(a, b, x=None, tol=0.01, max_iterations=100):
    n = len(b)
    if x is None:
        x = np.zeros_like(b)
    
    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(a[i][j] * x_new[j] for j in range(i))
            s2 = sum(a[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / a[i][i]
        
        if np.allclose(x, x_new, atol=tol):
            break
        
        x = x_new
    
    return x

a = np.array([[18, 8, -3, 4],
              [-7, 11, -5, 2],
              [4, 1, 3, 4],
              [-8, -8, -6, 31]], float)

b = np.array([-84, -5, -38, 263], float)

solution = seidel(a, b)
print("Решение методом Зейделя:", solution)
