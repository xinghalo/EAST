import numpy as np

p1 = np.array([0, 0])
p2 = np.array([2, 0])
p3 = np.array([1, 1])
print(np.cross(p2 - p1, p1 - p3))
print(np.linalg.norm(np.cross(p2 - p1, p1 - p3)))
print(np.linalg.norm(p2 - p1))
d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
print(d)