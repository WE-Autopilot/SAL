import torch
import torch.nn as nn


class PenalizeModel(nn.Module):
        def __init__(self, dev_weight=1.0, collision_weight=10.0, distance_weight=1.0, time_weight=0.1):
                super(PenalizeModel, self).__init__()

                # weight for mean dev penalty
                self.dev_weight = dev_weight

                # weight for collision penalty
                self.collision_weight = collision_weight

                # weight for final distance penalty
                self.distance_weight = distance_weight

                # weight for time penalty
                self.time_weight = time_weight


        def forward(self, predicted_path, reference_path, collision_flag, final_distance, time_spent):

                # compute dev at each step
                deviation = (predicted_path - reference_path).norm(p=2, dim=-1)

                # compute mean dev across all steps
                mean_deviation = deviation.mean()

                # apply penalty if collision occurs
                collision_penalty = collision_flag.float().mean()

                # penalize based on final distance to target
                distance_penalty = final_distance.mean()

                # penalize based on time taken to complete
                time_penalty = time_spent.mean()

                # total penalties at the end
                total_penalty = (self.dev_weight * mean_deviation
                                + self.collision_weight * collision_penalty
                                + self.distance_weight * distance_penalty
                                + self.time_weight * time_penalty)

                return total_penalty
