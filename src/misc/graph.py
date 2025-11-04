import torch
from torchviz import make_dot

from transformer.model import Transformer

model = Transformer(vocab_size=10000, d_model=512, seq_len=100)
dot = make_dot(model(torch.randint(0, 10000, (1, 100), dtype=torch.long), torch.randint(0, 10000, (1, 100), dtype=torch.long)), params=dict(model.named_parameters()))
dot.render("graph/transformer_graph", format="png")
