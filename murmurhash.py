import numpy as np
arr = np.arange(9).reshape((3, 3))
print(arr)
i = 0
while(i<3):
    ar= np.random.permutation(arr)
    i = i+1
    if ar.all == arr.all:
        print('tekrari')
        i = i-1
    print('iter:',i,ar)
