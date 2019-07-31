import numpy as np
import math

# prints formatted price
def formatPrice(n):
	return  ("-" if n < 0 else "")+ "{0}".format(format(abs(n), ',')) + "원" 

# returns the vector containing stock data from a fixed file
def getStockDataVec(key, tgtposi):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	# 제목 열 날리고 나머지 데이터들을 loop
	for line in lines[1:]:
		# [..., close_t-1, close_t, ... ] 
		vec.append(float(line.split(",")[tgtposi]))

	return vec

# returns the sigmoid
def sigmoid(x):
	try:
		retval = 1.0 / (1+ math.exp(-x))
	except OverflowError:
		retval = 1.0 / float('inf')
	
	
	return retval

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])
