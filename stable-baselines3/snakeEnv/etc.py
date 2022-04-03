import random
from scipy.spatial import distance

pos_1 = [100, 50]
target = [random.randrange(1, (480 // 10)) * 10, 
      random.randrange(1, (360 // 10)) * 10]

pos_2 = [110, 50]

print(pos_1)
print(target)
dist_1 = distance.euclidean(pos_1, target)
print(dist_1)

print(pos_2)
dist_2 = distance.euclidean(pos_2, target)
print(dist_2)

print(dist_1 - dist_2)