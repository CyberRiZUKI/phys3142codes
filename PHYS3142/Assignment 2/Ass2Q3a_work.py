import numpy as np
import random
import time

num_test = 10000
num_pos  = 6
pos_take = [0, 2, 4]

# ORIGINAL
print("\n[ORIGINAL]")
np.random.seed(2025)
start_time = time.perf_counter()

num_lose = 0
for nt in range(num_test):
    A = np.zeros(num_pos, bool)
    A[np.random.randint(0, num_pos)] = True
    for n in range(num_pos):
        if A[n] == 1:
            if n in pos_take:
                num_lose += 1
            break

original_time = time.perf_counter() - start_time
print(f"  Lose probability : {num_lose / num_test:.4f}")
print(f"  Time             : {original_time:.6f} sec")

# OPTIMIZED (np.random.randint)
print("\n[OPTIMIZED] np.bincount + np.random.randint")
np.random.seed(2025)
start_time = time.perf_counter()

bullet_positions = np.random.randint(0, num_pos, size=num_test)
counts = np.bincount(bullet_positions, minlength=num_pos)
num_lose_fast = np.sum(counts[pos_take])

fast_time_np = time.perf_counter() - start_time
print(f"  Lose probability : {num_lose_fast / num_test:.4f}")
print(f"  Time             : {fast_time_np:.6f} sec")
print(f"  Speedup          : {original_time / fast_time_np:.1f}x faster")

# OPTIMIZED (random.randint)
print("\n[OPTIMIZED] np.bincount + random.randint")
random.seed(2025)
start_time = time.perf_counter()

bullet_positions_r = [random.randint(0, num_pos - 1) for _ in range(num_test)]
counts_r = np.bincount(bullet_positions_r, minlength=num_pos)
num_lose_r = np.sum(counts_r[pos_take])

fast_time_r = time.perf_counter() - start_time
print(f"  Lose probability : {num_lose_r / num_test:.4f}")
print(f"  Time             : {fast_time_r:.6f} sec")
print(f"  Speedup          : {original_time / fast_time_r:.1f}x faster")

