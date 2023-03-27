import numpy as np
from numpy import linalg
a= [[1, 1, -1],[4,-1,5],[6,1,1]]
a1=[[7, 1, -1],[4,-1,5],[18,1,1]]
a2=[[1, 7, -1],[4,4,5],[6,18,1]]
a3=[[1, 1, 7],[4,-1,4],[6,1,18]]
# print(np.linalg.det(a))
x1=np.linalg.det(a1)/np.linalg.det(a)
x2=np.linalg.det(a2)/np.linalg.det(a)
x3=np.linalg.det(a3)/np.linalg.det(a)
print('x1',x1)
print('x2',x2)
print('x3',x3)
b = np.array([7, 4, 18])
x = np.linalg.solve(a, b)
print(x)