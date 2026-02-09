# Models Module

Neural network architectures for structural estimation.

## Network Classes

### StructuralNet

The main neural network architecture for structural parameter estimation.

```python
from deep_inference.models import StructuralNet

# Create network
net = StructuralNet(
    input_dim=10,           # Number of covariates
    hidden_dims=[64, 32],   # Hidden layer sizes
    theta_dim=2,            # Number of parameters (alpha, beta)
    dropout=0.1             # Dropout rate
)

# Forward pass
import torch
X = torch.randn(100, 10)
theta = net(X)  # (100, 2)
```

## Network Architecture

```
Input (d features)
    |
Linear(d, hidden_dims[0])
    |
ReLU + Dropout
    |
Linear(hidden_dims[0], hidden_dims[1])
    |
ReLU + Dropout
    |
...
    |
Linear(hidden_dims[-1], theta_dim)
    |
Output (theta_dim parameters per observation)
```

## Usage with structural_dml

The `structural_dml` function creates and trains the network internally:

```python
from deep_inference import structural_dml

result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',
    hidden_dims=[64, 32],  # Network architecture
    epochs=100,            # Training epochs
    lr=0.01               # Learning rate
)

# Access estimated parameters
theta_hat = result.theta_hat  # (n, theta_dim) numpy array
alpha_hat = theta_hat[:, 0]
beta_hat = theta_hat[:, 1]
```

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | `[64, 32]` | Hidden layer sizes |
| `epochs` | `100` | Training epochs |
| `lr` | `0.01` | Learning rate |
| `batch_size` | `64` | Mini-batch size |
| `weight_decay` | `1e-4` | L2 regularization |
| `dropout` | `0.1` | Dropout rate |

## Architecture Guidelines

| Sample Size | Recommended Architecture |
|-------------|-------------------------|
| n < 1,000 | `[32, 16]` |
| 1,000 < n < 10,000 | `[64, 32]` |
| 10,000 < n < 100,000 | `[128, 64, 32]` |
| n > 100,000 | `[256, 128, 64]` |

### MultinomialLogitModel

For multinomial logit (conditional logit / McFadden) choice models:

```python
from deep_inference.models.multinomial import MultinomialLogitModel

model = MultinomialLogitModel(n_alternatives=3, n_attributes=2)
# theta_dim = (J-1) + K = 4
# theta = [alpha_1, ..., alpha_{J-1}, beta_1, ..., beta_K]
# Hessian: Fisher information (no Y dependence, depends on theta)
# Requires 3-way cross-fitting
```

### CombinatorialModel

For multi-treatment combinatorial experiments with binary treatment vectors T ∈ {0,1}^m.

```python
from deep_inference.models.combinatorial import CombinatorialModel

model = CombinatorialModel(n_treatments=3, link='gen_sigmoid_ii')
# theta_dim = m + 2 = 5 for gen_sigmoid_ii
# theta = [θ₀, θ₁, θ₂, θ₃, θ₄]
```

**Four link functions:**

| Link | Formula | θ_dim | Description |
|------|---------|-------|-------------|
| `multiplicative` | θ₀ · ∏(1 + θ_k · t_k) | m+1 | Product interaction |
| `sigmoid` | a/(1+exp(-(θ₀ + Σ θ_k·t_k))) + b | m+1 | Bounded response with fixed scale |
| `gen_sigmoid_i` | θ_{m+1} · σ(Σ θ_k·t_k) | m+1 | Flexible scale, no intercept |
| `gen_sigmoid_ii` | θ_{m+1} · σ(θ₀ + Σ θ_k·t_k) | m+2 | Most flexible (recommended) |

**Hessian properties:**
- Uses Fisher information: 2·G_θ·G_θ' (does NOT depend on y)
- `hessian_depends_on_theta = True` (G_θ depends on θ)
- `hessian_depends_on_y = False` (Fisher information)
- Requires 3-way cross-fitting (Regime C)

**Lambda computation for randomized experiments:**

```python
# Compute Λ(x) via Monte Carlo for Regime A
import torch
t_samples = torch.randint(0, 2, (1000, 3)).float()
Lambda = model.compute_lambda_integral(theta, t_samples)
```

**Usage with MultiTreatmentATE:**

```python
from deep_inference.targets import MultiTreatmentATE

model = CombinatorialModel(n_treatments=3, link='gen_sigmoid_ii')
target = MultiTreatmentATE(model=model, treatment=[1, 0, 1])
```

*Reference: Ye et al. (2025, Management Science) — DeDL: Debiased Deep Learning for Combinatorial Experiments*

---

## Custom Network Usage

For advanced users who want to use the network directly:

```python
import torch
import torch.nn as nn
from deep_inference.models import StructuralNet
from deep_inference import LinearFamily

# Create network and family
net = StructuralNet(input_dim=10, hidden_dims=[64, 32], theta_dim=2)
family = LinearFamily()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    theta = net(X_tensor)
    loss = family.loss(Y_tensor, T_tensor, theta).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
