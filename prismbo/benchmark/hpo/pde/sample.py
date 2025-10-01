import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def random_sample_2d(num_x, num_y, range_x):

    gen = torch.Generator()
    gen.manual_seed(int(time.time() * 1e6) % (2**31 - 1))
    x = torch.rand(num_x, generator=gen) * (range_x[0, 1] - range_x[0, 0]) + range_x[0, 0]
    y = torch.rand(num_y, generator=gen) * (range_x[1, 1] - range_x[1, 0]) + range_x[1, 0]
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    return points


def get_data_with_mu(mu, num_x, num_y, num_bc, num_ic, range):
    """
    Generate 2D grid points and parameter mu values for PDE problems, for a given mu (single value).
    All outputs are in N/M format (no batch dimension).

    Args:
        mu (float or torch.Tensor): The parameter value to use (scalar or shape (1,))
        num_x (int): Number of grid points along x-axis.
        num_y (int): Number of grid points along y-axis.
        num_bc (int): Number of boundary points.
        num_ic (int): Number of initial points.
        range (torch.Tensor): 2x2 tensor specifying the min and max for x and y, shape [[x_min, x_max], [y_min, y_max]].

    Returns:
        X_f (torch.Tensor): Interior points, shape (num_x*num_y, 2)
        mu_f (torch.Tensor): Corresponding mu values, shape (num_x*num_y, 1)
        X_b (torch.Tensor): Boundary points, shape (num_bc, 2)
        mu_b (torch.Tensor): Corresponding mu values for boundary, shape (num_bc, 1)
        X_i (torch.Tensor): Initial points, shape (num_ic, 2)
        mu_i (torch.Tensor): Corresponding mu values for initial, shape (num_ic, 1)
    """
    x_lo, x_hi = range[0, 0], range[0, 1]
    y_lo, y_hi = range[1, 0], range[1, 1]

    # Create grid for (x, y)
    x = torch.linspace(x_lo, x_hi, num_x)
    y = torch.linspace(y_lo, y_hi, num_y)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (num_x*num_y, 2)

    # Interior points
    X_f = points  # (num_x*num_y, 2)
    mu_val = mu if isinstance(mu, torch.Tensor) else torch.tensor([mu], dtype=points.dtype, device=points.device)
    mu_f = mu_val.expand(points.shape[0], 1)  # (num_x*num_y, 1)

    # Boundary points: sample num_bc points uniformly along the boundary
    bc_points = []
    num_edges = 4
    per_edge = num_bc // num_edges
    remainder = num_bc % num_edges

    # Edge 1: x = x_lo, y in [y_lo, y_hi]
    n1 = per_edge + (1 if remainder > 0 else 0)
    y1 = torch.linspace(y_lo, y_hi, n1)
    edge1 = torch.stack([torch.full((n1,), x_lo, dtype=points.dtype, device=points.device), y1], dim=1)
    bc_points.append(edge1)

    # Edge 2: x = x_hi, y in [y_lo, y_hi]
    n2 = per_edge + (1 if remainder > 1 else 0)
    y2 = torch.linspace(y_lo, y_hi, n2)
    edge2 = torch.stack([torch.full((n2,), x_hi, dtype=points.dtype, device=points.device), y2], dim=1)
    bc_points.append(edge2)

    # Edge 3: y = y_lo, x in [x_lo, x_hi]
    n3 = per_edge + (1 if remainder > 2 else 0)
    x3 = torch.linspace(x_lo, x_hi, n3)
    edge3 = torch.stack([x3, torch.full((n3,), y_lo, dtype=points.dtype, device=points.device)], dim=1)
    bc_points.append(edge3)

    # Edge 4: y = y_hi, x in [x_lo, x_hi]
    n4 = per_edge
    x4 = torch.linspace(x_lo, x_hi, n4)
    edge4 = torch.stack([x4, torch.full((n4,), y_hi, dtype=points.dtype, device=points.device)], dim=1)
    bc_points.append(edge4)

    bc_points = torch.cat(bc_points, dim=0)[:num_bc]  # (num_bc, 2)
    X_b = bc_points
    mu_b = mu_val.expand(X_b.shape[0], 1)  # (num_bc, 1)

    # Initial points: y = y_lo, x in [x_lo, x_hi], sample num_ic points
    x_ic = torch.linspace(x_lo, x_hi, num_ic)
    ic_points = torch.stack([x_ic, torch.full((num_ic,), y_lo, dtype=points.dtype, device=points.device)], dim=1)  # (num_ic, 2)
    X_i = ic_points
    mu_i = mu_val.expand(X_i.shape[0], 1)  # (num_ic, 1)

    return X_f, mu_f, X_b, mu_b, X_i, mu_i


def get_data(num_mu, num_x, num_y, num_bc, num_ic, range, range_mu):
    """
    Generate 2D grid points and parameter mu values for PDE problems.
    All outputs are in B*N*M format, where B=num_mu (batch), N/M=number of points.

    Args:
        num_x (int): Number of grid points along x-axis.
        num_y (int): Number of grid points along y-axis.
        num_mu (int): Number of mu parameter samples.
        num_bc (int): Number of boundary points per mu.
        num_ic (int): Number of initial points per mu.
        range (torch.Tensor): 2x2 tensor specifying the min and max for x and y, shape [[x_min, x_max], [y_min, y_max]].
        range_mu (tuple or list or torch.Tensor): (mu_min, mu_max)

    Returns:
        X_f (torch.Tensor): Interior points, shape (num_mu, num_x*num_y, 2)
        mu_f (torch.Tensor): Corresponding mu values, shape (num_mu, num_x*num_y, 1)
        X_b (torch.Tensor): Boundary points, shape (num_mu, num_bc, 2)
        mu_b (torch.Tensor): Corresponding mu values for boundary, shape (num_mu, num_bc, 1)
        X_i (torch.Tensor): Initial points, shape (num_mu, num_ic, 2)
        mu_i (torch.Tensor): Corresponding mu values for initial, shape (num_mu, num_ic, 1)
        nus (torch.Tensor): All sampled mu values, shape (num_mu, 1)
    """
    x_lo, x_hi = range[0, 0], range[0, 1]
    y_lo, y_hi = range[1, 0], range[1, 1]
    mu_lo, mu_hi = range_mu[0], range_mu[1]

    # Sample mu values
    nus = torch.linspace(mu_lo, mu_hi, num_mu).unsqueeze(1)  # (num_mu, 1)

    # Create grid for (x, y)
    x = torch.linspace(x_lo, x_hi, num_x)
    y = torch.linspace(y_lo, y_hi, num_y)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (num_x*num_y, 2)

    # Interior points: all (x, y) for each mu, shape (num_mu, num_x*num_y, 2)
    X_f = points.unsqueeze(0).repeat(num_mu, 1, 1)  # (num_mu, num_x*num_y, 2)
    mu_f = nus.unsqueeze(1).repeat(1, num_x*num_y, 1)  # (num_mu, num_x*num_y, 1)

    # Boundary points: sample num_bc points uniformly along the boundary
    # Four edges: x = x_lo, x = x_hi, y = y_lo, y = y_hi
    # We'll sample num_bc points in total, distributed among the four edges
    bc_points = []
    num_edges = 4
    per_edge = num_bc // num_edges
    remainder = num_bc % num_edges

    # Edge 1: x = x_lo, y in [y_lo, y_hi]
    n1 = per_edge + (1 if remainder > 0 else 0)
    y1 = torch.linspace(y_lo, y_hi, n1)
    edge1 = torch.stack([torch.full((n1,), x_lo), y1], dim=1)
    bc_points.append(edge1)

    # Edge 2: x = x_hi, y in [y_lo, y_hi]
    n2 = per_edge + (1 if remainder > 1 else 0)
    y2 = torch.linspace(y_lo, y_hi, n2)
    edge2 = torch.stack([torch.full((n2,), x_hi), y2], dim=1)
    bc_points.append(edge2)

    # Edge 3: y = y_lo, x in [x_lo, x_hi]
    n3 = per_edge + (1 if remainder > 2 else 0)
    x3 = torch.linspace(x_lo, x_hi, n3)
    edge3 = torch.stack([x3, torch.full((n3,), y_lo)], dim=1)
    bc_points.append(edge3)

    # Edge 4: y = y_hi, x in [x_lo, x_hi]
    n4 = per_edge
    x4 = torch.linspace(x_lo, x_hi, n4)
    edge4 = torch.stack([x4, torch.full((n4,), y_hi)], dim=1)
    bc_points.append(edge4)

    bc_points = torch.cat(bc_points, dim=0)[:num_bc]  # (num_bc, 2)
    X_b = bc_points.unsqueeze(0).repeat(num_mu, 1, 1)  # (num_mu, num_bc, 2)
    mu_b = nus.unsqueeze(1).repeat(1, num_bc, 1)  # (num_mu, num_bc, 1)

    # Initial points: y = y_lo, x in [x_lo, x_hi], sample num_ic points
    x_ic = torch.linspace(x_lo, x_hi, num_ic)
    ic_points = torch.stack([x_ic, torch.full((num_ic,), y_lo)], dim=1)  # (num_ic, 2)
    X_i = ic_points.unsqueeze(0).repeat(num_mu, 1, 1)  # (num_mu, num_ic, 2)
    mu_i = nus.unsqueeze(1).repeat(1, num_ic, 1)  # (num_mu, num_ic, 1)

    return X_f, mu_f, X_b, mu_b, X_i, mu_i, nus

def get_data_2d(num_x, num_y, range_x):
    """
    Generate 2D grid points and improved boundary points for PDE problems.

    Args:
        num_x (int): Number of grid points along x-axis.
        num_y (int): Number of grid points along y-axis.
        range_x (torch.Tensor): 2x2 tensor specifying the min and max for x and y, shape [[x_min, x_max], [y_min, y_max]].

    Returns:
        points (torch.Tensor): All grid points, shape (num_x*num_y, 2)
        bc_points (torch.Tensor): Unique boundary points, shape (num_bc, 2)
        ic_points (list): Empty list (for compatibility)
    """
    x = torch.linspace(range_x[0, 0], range_x[0, 1], num_x)
    y = torch.linspace(range_x[1, 0], range_x[1, 1], num_y)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    # Improved boundary points: take all points on the boundary (edges), but avoid duplicates at corners
    # Four edges: x = x_min, x = x_max, y = y_min, y = y_max
    # 1. x = x_min, y in [y_min, y_max]
    left = torch.stack([torch.full((num_y,), range_x[0, 0]), y], dim=1)
    # 2. x = x_max, y in [y_min, y_max]
    right = torch.stack([torch.full((num_y,), range_x[0, 1]), y], dim=1)
    # 3. y = y_min, x in (x_min, x_max) (exclude corners)
    ic_points = torch.stack([x[1:-1], torch.full((num_x-2,), range_x[1, 0])], dim=1) if num_x > 2 else torch.empty((0,2))

    

    bc_points = torch.cat([left, right], dim=0)

    return points, bc_points, ic_points


def get_data_3d(num_x, num_y, num_z, range_x, device):
    # Generate 3D grid points
    x = torch.linspace(range_x[0, 0], range_x[0, 1], num_x)
    y = torch.linspace(range_x[1, 0], range_x[1, 1], num_y)
    z = torch.linspace(range_x[2, 0], range_x[2, 1], num_z)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([grid_x.reshape(-1),
                          grid_y.reshape(-1),
                          grid_z.reshape(-1)], dim=1)

    # Generate boundary points (6 faces for 3D cube)
    # Face 1: x = x_min
    y_b1, z_b1 = torch.meshgrid(y, z, indexing='ij')
    b1 = torch.stack([torch.full_like(y_b1.reshape(-1), range_x[0, 0]),
                      y_b1.reshape(-1),
                      z_b1.reshape(-1)], dim=1)

    # Face 2: x = x_max
    y_b2, z_b2 = torch.meshgrid(y, z, indexing='ij')
    b2 = torch.stack([torch.full_like(y_b2.reshape(-1), range_x[0, 1]),
                      y_b2.reshape(-1),
                      z_b2.reshape(-1)], dim=1)

    # Face 3: y = y_min
    x_b3, z_b3 = torch.meshgrid(x, z, indexing='ij')
    b3 = torch.stack([x_b3.reshape(-1),
                      torch.full_like(x_b3.reshape(-1), range_x[1, 0]),
                      z_b3.reshape(-1)], dim=1)

    # Face 4: y = y_max
    x_b4, z_b4 = torch.meshgrid(x, z, indexing='ij')
    b4 = torch.stack([x_b4.reshape(-1),
                      torch.full_like(x_b4.reshape(-1), range_x[1, 1]),
                      z_b4.reshape(-1)], dim=1)

    # Face 5: z = z_min
    x_b5, y_b5 = torch.meshgrid(x, y, indexing='ij')
    b5 = torch.stack([x_b5.reshape(-1),
                      y_b5.reshape(-1),
                      torch.full_like(x_b5.reshape(-1), range_x[2, 0])], dim=1)

    # Face 6: z = z_max
    x_b6, y_b6 = torch.meshgrid(x, y, indexing='ij')
    b6 = torch.stack([x_b6.reshape(-1),
                      y_b6.reshape(-1),
                      torch.full_like(x_b6.reshape(-1), range_x[2, 1])], dim=1)

    # Move all tensors to specified device
    points = points.to(device)
    b1, b2, b3, b4, b5, b6 = [b.to(device) for b in [b1, b2, b3, b4, b5, b6]]

    return points, b1, b2, b3, b4, b5, b6


def plot_func_2d(points_test, model, func, range_x, save_path=None, test_name="result"):

    # Automatically determine grid size from points
    num_points = points_test.shape[0]
    grid_size = int(np.sqrt(num_points))

    if grid_size * grid_size != num_points:
        raise ValueError(f"Number of test points ({num_points}) must be a perfect square for grid plotting")

    # Convert range_x to CPU numpy array
    range_x_np = range_x.cpu().numpy() if hasattr(range_x, 'cpu') else range_x

    # Get predictions and true values
    model.eval()
    with torch.no_grad():
        y_pred = model(points_test).cpu().numpy()
    y_true = func(points_test).cpu().numpy()

    # Reshape for plotting
    x = points_test[:, 0].cpu().numpy().reshape(grid_size, grid_size)
    t = points_test[:, 1].cpu().numpy().reshape(grid_size, grid_size)
    z_true = y_true.reshape(grid_size, grid_size)
    z_pred = y_pred.reshape(grid_size, grid_size)
    z_err = np.abs(z_true - z_pred)

    # Calculate error statistics
    max_error = np.max(z_err)
    mean_error = np.mean(z_err)
    l2_error = np.sqrt(np.mean(z_err ** 2))

    # Plot 1: Predicted physical field
    plt.figure(figsize=(5, 4))
    plt.pcolormesh(x, t, z_pred, cmap='viridis', shading='gouraud')
    plt.title('Predicted Field', fontsize=15)
    plt.colorbar()
    plt.xlabel('x', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.xlim(range_x_np[0][0], range_x_np[0][1])
    plt.ylim(range_x_np[1][0], range_x_np[1][1])
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{test_name}_predicted.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: True physical field
    plt.figure(figsize=(5, 4))
    plt.pcolormesh(x, t, z_true, cmap='viridis', shading='gouraud')
    plt.title('Ground Truth', fontsize=15)
    plt.colorbar()
    plt.xlabel('x', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.xlim(range_x_np[0][0], range_x_np[0][1])
    plt.ylim(range_x_np[1][0], range_x_np[1][1])
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{test_name}_ground_truth.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 3: Error field
    plt.figure(figsize=(5, 4))
    plt.pcolormesh(x, t, z_err, cmap='Reds', shading='gouraud')
    plt.title('Absolute Error', fontsize=15)
    plt.colorbar()
    plt.xlabel('x', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.xlim(range_x_np[0][0], range_x_np[0][1])
    plt.ylim(range_x_np[1][0], range_x_np[1][1])
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{test_name}_error.png", dpi=300, bbox_inches='tight')
    plt.show()

