from math import gcd

def c2n(c):
    return ord(c) - 97

def n2c(n):
    return chr(n + 97)

# ── Find Multiplicative Inverses in Z/26
print("Multiplicative Inverses in Z/26")
print("")
print(f"{'a':>4} | {'a⁻¹':>4} | {'Verification (a⁻¹ × a) mod 26':>30}")
print("")

count = 0
inverses = {}

for a in range(26):
    found = False
    for b in range(26):
        if (a * b) % 26 == 1:
            inverses[a] = b
            print(f"{a:>4} | {b:>4} | {b} × {a} = {b * a} ≡ {(b * a) % 26} (mod 26)")
            count += 1
            found = True
            break  # Only need one inverse
    if not found:
        print(f"{a:>4} | {'None':>4} | No inverse exists")

print("")
print(f"\nTotal elements with a multiplicative inverse: {count}")

# Show the pattern: GCD condition
print("Check GCD(a, 26) for each element:")
print("-" * 40)
print(f"{'a':>4} | {'gcd(a,26)':>10} | {'Has Inverse?':>12}")
print("-" * 40)

for a in range(26):
    g = gcd(a, 26)
    has_inv = "Yes" if a in inverses else "No"
    print(f"{a:>4} | {g:>10} | {has_inv:>12}")

# Confirm Affine Cipher Encryption & Decryption, Part b
a = 7
b = 18
x = 3  #given

# Step 1: Encrypt  →  y = (a × x + b) mod 26
y = (a * x + b) % 26
print(f"y = {y}")
print(f"Confirmed: y = {y} (expected 13)" if y == 13 else f"GG")

# Step 2: Decrypt  →  x = a⁻¹ × (y − b) mod 26
a_inv = inverses[a]
x_decrypted = (a_inv * (y - b)) % 26
print(f"x = {x_decrypted}")
print(f"Confirmed: x = {x_decrypted} (expected 3)" if x_decrypted == 3 else f"Dame")

# Step 3: Summary
print(f"\nSummary")
print(f"Plaintext  x = {x}")
print(f"ENcrypted  y = {y}")
print(f"DEcrypted  x = {x_decrypted}")
print(f"\nThe affine cipher is fully reversible.")


# Part C
# Find all multiplicative inverses in Z/26
inverses = {}
for a in range(26):
    for b in range(26):
        if (a * b) % 26 == 1:
            inverses[a] = b
            break

# ── Encrypted message given in homework1
ciphertext = "gkstmdodikbojsydzkpuibtzwuigu"

# Convert ciphertext to numbers
cipher_numbers = [c2n(c) for c in ciphertext]
print(f"\nCiphertext: {ciphertext}")
print(f"As numbers: {cipher_numbers}\n")

# Brute-force all valid (a, b) combinations
# Constraint: first two plaintext letters are 'c' and 'o' (2 and 14)

target_first = c2n('c')   # 2
target_second = c2n('o')  # 14
solutions = []

for a in inverses:          # Only try a values that have inverses
    a_inv = inverses[a]
    for b in range(26):
        x0 = (a_inv * (cipher_numbers[0] - b)) % 26
        x1 = (a_inv * (cipher_numbers[1] - b)) % 26

        if x0 == target_first and x1 == target_second:
            plaintext = ""
            for y in cipher_numbers:
                x = (a_inv * (y - b)) % 26
                plaintext += n2c(x)

            solutions.append((a, b, a_inv, plaintext))
            print(f"a = {a}, b = {b}, a⁻¹ = {a_inv}")
            print(f"Decrypted message: {plaintext}")

print(f"SOLUTIONS FOUND: {len(solutions)}")

