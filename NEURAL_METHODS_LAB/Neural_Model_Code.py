# ===== Task 3: binary XOR (LLM-generated, tanh 2-2-1) =====
import torch, torch.nn as nn

torch.manual_seed(0)  # reproducibility

# The four XOR training examples, explicitly
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# 2-2-1 network: tanh hidden activation, one output logit, random init (PyTorch default)
model = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1))
loss_fn = nn.BCEWithLogitsLoss()          # sigmoid + binary cross-entropy, numerically stable
opt = torch.optim.Adam(model.parameters(), lr=0.05)

# Full-batch training: all 4 examples every step, 3000 CPU steps
for step in range(3000):
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()

# Fresh forward/backward pass at the trained weights so the gradient is exposed
opt.zero_grad()
loss = loss_fn(model(X), y)
loss.backward()
print("Final loss:", loss.item())

with torch.no_grad():
    probs = torch.sigmoid(model(X))       # logits -> probabilities
print("Probabilities:", probs.squeeze().tolist())
print("Labels (p>0.5):", (probs > 0.5).int().squeeze().tolist())
print("Grad of hidden weight (model[0].weight.grad):\n", model[0].weight.grad)


# ===== Task 4: Parts A-D, seed check, diagnostic =====
import torch, torch.nn as nn

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])
loss_fn = nn.BCEWithLogitsLoss()

def make_model(act):
    return nn.Sequential(nn.Linear(2, 2), act, nn.Linear(2, 1))

def train(model, steps=3000, lr=0.05, log_steps=()):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    first_loss, early_norm = None, None
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        if step == 0:
            first_loss = loss.item()
            early_norm = model[0].weight.grad.norm().item()   # early ||dL/dW1||_2
        if step in log_steps:
            print(f"step {step}: loss={loss.item():.4f}\n", model[0].weight.data)
        opt.step()
    with torch.no_grad():
        p = torch.sigmoid(model(X)).squeeze()
    final_loss = loss_fn(model(X), y).item()
    return first_loss, final_loss, p, (p > 0.5).int(), early_norm

# ---- Part A: basic learning check ----
torch.manual_seed(0)
first, final, p, lab, _ = train(make_model(nn.Tanh()))
print("A: initial loss", first, "final loss", final)
print("A: probabilities", p.tolist(), "labels", lab.tolist())

# ---- Part B: gradient of first-layer weights ----
torch.manual_seed(0)
m = make_model(nn.Tanh())
m.zero_grad()
loss_fn(m(X), y).backward()
print("B: model[0].weight.grad = dL/dW1\n", m[0].weight.grad)

# ---- Part C: symmetry experiment (all weights and biases zero) ----
torch.manual_seed(0)
m = make_model(nn.Tanh())
for prm in m.parameters():
    nn.init.zeros_(prm)
train(m, steps=1000, log_steps=(0, 1, 10, 100, 999))
print("C: rows identical?", torch.equal(m[0].weight[0], m[0].weight[1]))

# ---- Part D: activation experiment ----
print("D: activation | final loss | 4/4 correct | early grad norm")
for name, act in [("sigmoid", nn.Sigmoid), ("tanh", nn.Tanh), ("relu", nn.ReLU)]:
    torch.manual_seed(0)
    _, final, p, lab, g = train(make_model(act()))
    ok = lab.tolist() == [0, 1, 1, 0]
    print(f"{name:8s} {final:.5f} {ok} {g:.5f}")

# ---- Repeated-run check (validation criterion) ----
for name, act in [("sigmoid", nn.Sigmoid), ("tanh", nn.Tanh), ("relu", nn.ReLU)]:
    wins = 0
    for seed in range(10):
        torch.manual_seed(seed)
        _, _, _, lab, _ = train(make_model(act()))
        wins += lab.tolist() == [0, 1, 1, 0]
    print(name, "solved", wins, "/ 10 seeds")

# ---- Diagnostic: saturated sigmoid vs dead ReLU (seed 0) ----
for name, act in [("sigmoid", nn.Sigmoid), ("relu", nn.ReLU)]:
    torch.manual_seed(0)
    m = make_model(act())
    train(m)
    with torch.no_grad():
        a = m[0](X)       # pre-activations
        h = m[1](a)       # hidden outputs
    print(name, "pre-activations:\n", a, "\nhidden outputs:\n", h)


# ===== Task 5: three-class extension =====
import torch, torch.nn as nn

torch.manual_seed(0)
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0, 1, 1, 2])               # integer class labels

# Only the output layer and loss change: 3 logits, CrossEntropyLoss (takes raw logits)
model = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 3))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(3000):
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()

print("Final loss:", loss.item())
print("Output weight shape:", model[2].weight.shape)    # expect (3, 2)

with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=1)
print("Probabilities:\n", probs)
print("Predicted classes:", probs.argmax(dim=1).tolist())
print("Example 1 softmax:", probs[1].tolist(), "sum =", probs[1].sum().item())

# Optional: shift invariance and stability
shifted = torch.softmax(logits[1] + 100, dim=0)
print("Shifted by +100:", shifted.tolist())
print("Max abs difference:", (shifted - probs[1]).abs().max().item())
