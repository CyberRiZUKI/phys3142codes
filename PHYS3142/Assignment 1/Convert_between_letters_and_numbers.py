## This part is for converting between letters and numbers
## You can use the below functions or write your own code

# Convert a letter to its corresponding number (0 for 'a', 1 for 'b', ..., 25 for 'z')
def c2n(c):
    return ord(c) - 97

# Convert a number (0-25) back to its corresponding letter
def n2c(n):
    return chr(n + 97)


## This part is for showing the results of the above programs
## This part demonstrates the use of two different data types; you can also try using different data types.

message = "gkstmdodikbojsydzkpuibtzwuigu"
message_number = []                         # List type
message_new1 = ""                           # String type
message_new2 = ""

for i in range(len(message)):
    message_number.append(c2n(message[i]))  # Use append to add numbers to the list
print(message_number)

for i in range(len(message_number)):
    message_new1 += n2c(message_number[i])  # Use += to concatenate to the string
print(message_new1)

for i in message_number:
    message_new2 += n2c(i)
print(message_new2)
