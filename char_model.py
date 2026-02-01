import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_folder = "data"
text = ""

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
            text += f.read() + "\n"  # add newline between books

print("Length of text:", len(text))

# get unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

print("Unique characters:", vocab_size)

# create mappings
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# encode the entire text as numbers
encoded = [char_to_idx[ch] for ch in text]

# sanity checks
print("First 100 characters:", text[:100])
print("First 100 encoded numbers:", encoded[:100])

# context length (how many characters the model sees)
block_size = 16

x = []
y = []

for i in range(len(encoded) - block_size):
    x.append(encoded[i:i + block_size])
    y.append(encoded[i + block_size])

# inspect one example
sample_x = x[0]
sample_y = y[0]

print("Input (numbers):", sample_x)
print("Target (number):", sample_y)

print("Input (chars):", "".join(idx_to_char[i] for i in sample_x))
print("Target (char):", idx_to_char[sample_y])
