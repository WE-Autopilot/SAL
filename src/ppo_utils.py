import torch as pt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss
from tqdm import tqdm


# PPO update function remains the same.
def ppo_update(model, optimizer, images, positions, actions, log_probs_old, losses, advantages, clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01, epochs=4, mini_batch_size=16, shuffle=True):

    log_probs_old = log_probs_old.detach()
    losses = losses.detach()
    advantages = advantages.detach()


    dataset = TensorDataset(images, positions, actions, log_probs_old, losses, advantages)
    dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=shuffle)

    for _ in tqdm(range(epochs), desc="training...", unit="epoch"):
        for batch_images, batch_positions, batch_actions, batch_log_probs_old, batch_losses, batch_advantages in dataloader:
            dist, values = model(batch_images, batch_positions)
            
            batch_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            ratio = pt.exp(batch_log_probs - batch_log_probs_old).mean(dim=-1)
            surr1 = ratio * batch_advantages
            surr2 = pt.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
            policy_reward = pt.min(surr1, surr2).mean()
            
            value_loss = (batch_losses - values).pow(2).mean()
            loss = -policy_reward + value_loss_coef * value_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def angle_accel(vectors):
    """
    Calculates the average second-order difference of angles from a batch of 2D vectors.

    Args:
        vectors (pt.Tensor): Tensor of shape (4, 16, 2) representing 4 batches of 16 2D vectors.

    Returns:
        pt.Tensor: The average second-order difference of angles.
    """
    # 1. Calculate angles
    angles = pt.atan2(vectors[:, :, 1], vectors[:, :, 0]) # Shape (4, 16)

    # 2. Calculate the second-order difference
    second_order_diff = angles[:, 2:] - 2 * angles[:, 1:-1] + angles[:, :-2] # Shape (4, 14)

    # 3. Calculate the average of the second-order difference
    average_second_order_diff = pt.mean(pt.abs(second_order_diff))

    return average_second_order_diff
