from scipy.stats import pearsonr, spearmanr
import numpy as np 
import random 

a = random.sample(range(1, 50), 7)
b = random.sample(range(1, 50), 6)
print(a, b)
corr = spearmanr(a, b)
print("corr", corr)