"""
Here we use random number to simulate a game
@author: Junwei Liu
"""
import numpy as np
import time
#only one bullet in the gun
num_test=10000
num_pos=6
pos_take=[0,2,4]
start_time=time.time()
np.random.seed(2025)
num_lose=0
for nt in range(num_test):
	A=np.zeros(num_pos,bool)
	A[np.random.randint(0,num_pos)]=True
	#more precise simulations for the scenario in the video
	#since the first shot is empty
	# A[np.random.randint(1,num_pos)]=True
	##more elegent
	# if any(A[pos_take]):
	# num_lose += 1
	##easy to extend  
	for n in range(num_pos):
		if A[n]==1:
			if n in pos_take:
				num_lose += 1
			break
print("The lose probability is:", num_lose/num_test)
print("Time:",time.time()-start_time)



