import numpy as np
from scipy.spatial.distance import jensenshannon

before = np.array([0.9, 0.7, 0.3])
after  = np.array([0.8, 0.8, 0.8])

p = before / before.sum()
q = after  / after.sum()

js = jensenshannon(p, q)  # symmetric, bounded [0,1]
print("Jensen-Shannon Distance:", js)
