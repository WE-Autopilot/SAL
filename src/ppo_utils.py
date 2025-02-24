import torch as pt
import torch.nn.functional as F

def compute_entropy(probs):
    return -(probs * probs.log()).sum(dim=-1)

def ppo_update(actor, critic, actor_optimizer, critic_optimizer, states, actions, rewards, old_probs, clip_param=0.2, c1=0.5, c2=0.01):
    states = pt.stack(states)
    actions = pt.tensor(actions, dtype=pt.long)
    rewards = pt.tensor(rewards, dtype=pt.float32)
    old_probs = pt.stack(old_probs)

    # compute new action probabilities and state values
    new_probs = actor(states).gather(1, actions.unsqueeze(1)).squeeze()
    values = critic(states).squeeze()
    
    # compute advantages
    advantages = rewards - values.detach()

    # ratio (pi_theta / pi_theta_old)
    ratio = new_probs / old_probs

    # surrogate loss with clipping
    surr1 = ratio * advantages
    surr2 = pt.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
    actor_loss = -pt.min(surr1, surr2).mean()

    # critic loss (MSE between predicted and actual rewards)
    critic_loss = F.mse_loss(values, rewards)

    # entropy bonus
    entropy = compute_entropy(new_probs).mean()

    # PPO total loss
    loss = actor_loss - c1 * critic_loss + c2 * entropy

    # update actor and critic
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()
    critic_optimizer.step()  