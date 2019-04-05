import numpy as np

a = np.array([0, 1, 2, 3])
#print(type(a))
#print(a.dtype)
#print(a.shape)
# print(a)

b = np.array([[0,1.2,2,3], [4,5,6,7]])
#print(b.dtype)
#print(b)

c = np.array([[0,1.2,2,3], [4,5,6,7]])
#print(c.dtype)
#print(c)

d = np.ones((3,4))
print(d)
e = np.zeros((3,4))
print(e)
f = np.full((3,4),88)
print(f)

g = np.random.random_integers(5,size=(3,4))
print(g)
print(g[2,3])

h = np.eye(3,3)

a = np.array([1,2,3,4])
b = np.array([4,5,7,8])

sum = a+b
mult = a*b
expon = a**b