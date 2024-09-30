import gradio as gr

# from huggingface_hub import InferenceClient

import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
block_size = 8
max_iterations = 20000
learning_rate = 1e-3
eval_interval = 300
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

with open("potter.txt", "r", encoding="utf-8") as f:
    dataset = f.read()
    chars = sorted(list(set(dataset)))
    vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}  # string to index mapping
itos = {i: ch for i, ch in enumerate(chars)}  # index to string mapping
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])
data = torch.tensor(encode(dataset), dtype=torch.long)

# 90% training, 10% validation
n = int(0.9 * len(data))
training_data = data[:n]
validation_data = data[n:]


def get_batch(type):
    data = training_data if type == "training" else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        w = q @ k.transpose(-2, -1) * C**-0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.value(x)
        out = w @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["training", "validation"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramModel(vocab_size)
m = model.to(device)
# print(sum(p.numel() for p in m.parameters()) / 1e3, "K parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model, already trained (model.pth) so no need to train again
# for steps in range(max_iterations):
#     if steps % eval_interval == 0:
#         losses = estimate_loss()
#         print(
#             f"Step: {steps}, Training Loss: {losses['training']:.4f}, Validation Loss: {losses['validation']:.4f}"
#         )
#     xb, yb = get_batch("training")
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()


# prompt = str(input("Enter a prompt: "))
# context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


# load the model.pth
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

def respond(
    message,
    max_tokens=512,
):
    context = torch.tensor(encode(message), dtype=torch.long, device=device).unsqueeze(
        0
    )
    response = decode(model.generate(context, max_new_tokens=max_tokens)[0].tolist())
    return response


iface = gr.Interface(
    fn=respond,
    inputs=[
        gr.Textbox(lines=5, label="Message", value="Hi Harry Potter"),
        gr.Slider(minimum=100, maximum=2048, value=256, label="Max Tokens"),
    ],
    outputs="text",
    title="PotterLLM",
    description="A language model trained on Harry Potter Series.",
    theme="huggingface",
)

if __name__ == "__main__":
    iface.launch()
