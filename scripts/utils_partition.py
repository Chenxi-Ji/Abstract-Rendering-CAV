import os

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

def generate_partition(odd_type, part=None):
    if odd_type == "cylinder":
        if part is None:
            part = [1,1,1]
        input_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        input_max = np.array([1.0, 1.0, 6.28], dtype=np.float32)
        part = np.array(part)  # partitions per dimension

        # Step 1: compute edges per dimension
        edges = [
            np.linspace(input_min[d], input_max[d], part[d] + 1)
            for d in range(3)
        ]

        # Step 2: generate partitions
        centers = []
        lower_bounds = []
        upper_bounds = []

        for i in range(part[0]):
            for j in range(part[1]):
                for k in range(part[2]):
                    lb = np.array([
                        edges[0][i],
                        edges[1][j],
                        edges[2][k],
                    ])
                    ub = np.array([
                        edges[0][i + 1],
                        edges[1][j + 1],
                        edges[2][k + 1],
                    ])
                    center = (lb + ub) * 0.5

                    lower_bounds.append(lb)
                    upper_bounds.append(ub)
                    centers.append(center)

        # Convert to arrays
        lower_bounds = np.stack(lower_bounds)   # (250, 3)
        upper_bounds = np.stack(upper_bounds)   # (250, 3)
        centers = np.stack(centers)             # (250, 3)

        eps = 1e-8
        assert np.all(lower_bounds + eps < centers), "Some centers <= lower bounds!"
        assert np.all(upper_bounds - eps > centers), "Some centers >= upper bounds!"


        return centers, lower_bounds, upper_bounds
    
def refine_partition(input_center, input_lb, input_ub, odd_type, part=10000):

    eps = 1e-8
    assert input_center.shape == input_lb.shape == input_ub.shape
    assert torch.all(input_lb <= input_center+eps )
    assert torch.all(input_ub >= input_center-eps)

    #print(input_center,input_lb,input_ub)

    if odd_type == "cylinder":
        delta_lb = input_center - input_lb
        delta_ub = input_ub - input_center

        input_center[..., :2] = input_center[..., :2]/100
        delta_lb, delta_ub = delta_lb/part, delta_ub/part
        input_lb = input_center - delta_lb
        input_ub = input_center + delta_ub

    #print(input_center,input_lb,input_ub)

    return input_center, input_lb, input_ub


if __name__ == '__main__':
    odd_type = "cylinder"
    part = [1,5,10]
    
    centers, lower_bounds, upper_bounds = generate_partition(odd_type, part)
    print(lower_bounds.shape, upper_bounds.shape, centers.shape)
    # print(centers, lower_bounds, upper_bounds)