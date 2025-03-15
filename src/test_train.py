import torch as pt
from torch.optim import AdamW
from sal import SAL
from ppo_utils import ppo_update


sal = SAL()
optimizer = AdamW(sal.parameters(), lr=1e-4)

img = pt.randn(16, 1, 256, 256)
pos = pt.randn(16, 2)
rewards = pt.randn(16)

dist, value = sal(img, pos)
actions = dist.sample()
log_probs = dist.log_prob(actions)
advantages = rewards - value

ppo_update(sal, optimizer, img, pos, actions, log_probs.detach(), rewards.detach(), advantages.detach())
