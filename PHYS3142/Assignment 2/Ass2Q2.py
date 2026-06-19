import math

# (a) Function to Test if a Number is Prime
def is_prime(n):
    # Handle base cases
    if n < 2:
        return False        # 0 and 1 are not prime
    if n == 2:
        return True         # 2 is the only even prime
    if n % 2 == 0:
        return False        # eliminate all other even numbers

    # Only check odd divisors from 3 up to √n
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False

    return True

# (b) Find All Primes Smaller Than 1000
def sieve_of_eratosthenes(limit):
    is_prime_sieve = [True] * limit         # index = number, value = prime?
    is_prime_sieve[0] = False               # 0 is not prime
    is_prime_sieve[1] = False               # 1 is not prime

    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime_sieve[i]:               # if i is still marked prime
            for j in range(i*i, limit, i): # mark all multiples of i
                is_prime_sieve[j] = False

    # Collect all indices still marked True
    primes = [num for num, flag in enumerate(is_prime_sieve) if flag]
    return primes

#MAIN PROGRAM

print("PRIME NUMBER PROGRAM")

# Part (a): Test a single number
print("\n(a) PRIME NUMBER CHECKER")
# Interactive input
print()
user_input = int(input("Enter your own number to test: "))
if is_prime(user_input):
    print(f"  {user_input} is a PRIME number ✓")
else:
    print(f"  {user_input} is NOT a prime number ✗")

# Part (b): All primes under 1000
print("\n(b) ALL PRIME NUMBERS SMALLER THAN 1000")

primes_under_1000 = sieve_of_eratosthenes(1000)

# Print in a neat grid (10 per row)
print()
for i, p in enumerate(primes_under_1000):
    print(f"{p:>4}", end="  ")
    if (i + 1) % 10 == 0:       # new line every 10 numbers
        print()

print(f"\n\nTotal prime numbers smaller than 1000: {len(primes_under_1000)}")
