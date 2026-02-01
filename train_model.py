import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import pickle
from tqdm import tqdm

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


os.chdir(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_PATH = "shakespeare_checkpoint.pt"
VOCAB_PATH = "shakespeare_vocab.pkl"

# -----------------------
# Step 1: Load text
# -----------------------
data_folder = "data"
text_list = []

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
            text_list.append(f.read())

random.shuffle(text_list)
text = "\n".join(text_list)

print("Total raw characters:", len(text))

# -----------------------
# Step 2: Load or build vocab
# -----------------------
if os.path.exists(VOCAB_PATH):
    print("Loading saved vocabulary...")
    with open(VOCAB_PATH, "rb") as f:
        chars, char_to_idx, idx_to_char = pickle.load(f)
else:
    print("Building vocabulary...")
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump((chars, char_to_idx, idx_to_char), f)

vocab_size = len(chars)
encoded = [char_to_idx[ch] for ch in text]

print("Vocabulary size:", vocab_size)

# -----------------------
# Dataset statistics
# -----------------------
words = text.split()
unique_words = set(words)

print("\n--- Dataset stats ---")
print(f"Total characters      : {len(text)}")
print(f"Total words           : {len(words)}")
print(f"Unique words (approx) : {len(unique_words)}")
print(f"Unique characters     : {vocab_size}")



# -----------------------
# Step 3: Prepare training data
# -----------------------
block_size = 128
x_data, y_data = [], []

for i in range(len(encoded) - block_size):
    x_data.append(encoded[i:i + block_size])
    y_data.append(encoded[i + block_size])

X = torch.tensor(x_data, dtype=torch.long)
Y = torch.tensor(y_data, dtype=torch.long)

print(f"Block size            : {block_size}")

# -----------------------
# Step 4: Define model
# -----------------------
class CharModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=3, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

model = CharModel(vocab_size)

total_params, trainable_params = count_parameters(model)

print("\n--- Model configuration ---")
print(model)
print(f"\nTotal parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")
print(f"Embedding size       : {model.embed.embedding_dim}")
print(f"Hidden size          : {model.rnn.hidden_size}")
print(f"RNN layers           : {model.rnn.num_layers}")
print(f"Dropout              : {model.rnn.dropout}")


optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss()

print("\n--- Training configuration ---")
print(f"Optimizer  : Adam")
print(f"Learning rate : {optimizer.param_groups[0]['lr']}")
print(f"Betas      : {optimizer.param_groups[0]['betas']}")
print(f"Loss       : CrossEntropyLoss")



# -----------------------
# Step 5: Load checkpoint (MODEL + ADAM + EPOCH)
# -----------------------
start_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    print("Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1

    print(f"Resuming from epoch {start_epoch}")
    last_trained_epoch = start_epoch - 1

else:
    print("No checkpoint found. Training from scratch.")

# -----------------------
# Step 6: Train (REAL continuation)
# -----------------------
model.train()

batch_size = 256
epochs = 0  # Change from 0 to 15 (or whatever you want)

best_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(start_epoch, start_epoch + epochs):
    perm = torch.randperm(len(X))
    X = X[perm]
    Y = Y[perm]

    epoch_loss = 0.0
    progress = tqdm(
        range(0, len(X), batch_size),
        desc=f"Epoch {epoch}"
    )

    for i in progress:
        xb = X[i:i + batch_size]
        yb = Y[i:i + batch_size]

        optimizer.zero_grad()
        out, _ = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(progress)
    print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

    # 👇 ADD THESE LINES HERE (after printing avg_loss)
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        print(f"New best loss! ({best_loss:.4f})")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epoch(s)")
        if patience_counter >= patience:
            print("Loss plateaued, stopping early")
            break

    # 🔥 Save after every epoch
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, CHECKPOINT_PATH)

    print("Checkpoint saved.")
    last_trained_epoch = epoch

print("\n--- Training summary ---")
if epochs == 0:
    print("No training run in this session.")
    print(f"Batch size : {batch_size}")
    print(f"Patience   : {patience}")
else:
    print(f"Last trained epoch : {last_trained_epoch}")
    print(f"Best loss achieved : {best_loss:.4f}")
    print(f"Batch size : {batch_size}")
    print(f"Patience   : {patience}")


# -----------------------
# Step 7: Generate text
# -----------------------
model.eval()

num_samples = 5
gen_length = 300
temperature = 0.7

print("\n--- Generation configuration ---")
print(f"Temperature : {temperature}")
print(f"Generation length : {300}")


for sample_idx in range(num_samples):
    start_idx = random.randint(0, len(text) - block_size - 1)
    seed = text[start_idx:start_idx + block_size]

    input_seq = torch.tensor(
        [char_to_idx[ch] for ch in seed],
        dtype=torch.long
    ).unsqueeze(0)

    generated = seed
    hidden = None  # reset hidden state for each sample

    with torch.no_grad():
        for _ in range(gen_length):
            out, hidden = model(input_seq, hidden)
            probs = torch.softmax(out / temperature, dim=-1)
            char_idx = torch.multinomial(probs, 1).item()

            generated += idx_to_char[char_idx]
            input_seq = torch.tensor(
                [char_to_idx[ch] for ch in generated[-block_size:]],
                dtype=torch.long
            ).unsqueeze(0)

    print(f"\n--- Generated text {sample_idx + 1} ---\n")
    print(generated)