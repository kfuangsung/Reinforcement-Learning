import random
import numpy as np 
from scipy.spatial import distance

# pos_1 = [100, 50]
# target = [random.randrange(1, (480 // 10)) * 10, 
#       random.randrange(1, (360 // 10)) * 10]

# pos_2 = [110, 50]

# print(pos_1)
# print(target)
# dist_1 = distance.euclidean(pos_1, target)
# print(dist_1)

# print(pos_2)
# dist_2 = distance.euclidean(pos_2, target)
# print(dist_2)

# print(dist_1 - dist_2)

# q1 = int(np.round(np.quantile(range(20), 0.25), 0))
# mid = int(np.round(np.median(range(20)), 0))
# q3 = int(np.round(np.quantile(range(20), 0.75), 0))

# print(q1, mid, q3)

snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]
print(len(snake_body))