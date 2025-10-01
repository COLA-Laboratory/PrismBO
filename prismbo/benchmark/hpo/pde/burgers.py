import numpy as np
import torch
from prismbo.benchmark.hpo.pde.base import PDE
from typing import Callable, Optional, Tuple, Dict, Literal
import matplotlib.pyplot as plt


class Burgers2D(PDE):
    def __init__(self, seed: int = 42, mu: float = 0.01):
        self.seed = seed
        
    @property
    def range(self):
        return torch.tensor([[-1., 1.], [0., 1.]])
    
    @property
    def mu_range(self):
        return (0.005, 1.0)
    
    @property
    def num_mu(self):
        return 64
    
    def pde(self, x, y, mu):
        x.requires_grad_(True)
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        dy_t = dy[:, 1:2]
        dy_x = dy[:, 0:1]
        dy_xx = torch.autograd.grad(dy_x, x, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
        return dy_t + y * dy_x - mu * dy_xx
    
    def bc(self, x, y, mu):
        return torch.mean(y ** 2)
    
    def ic(self, x, y, mu):
        target = -torch.sin(torch.pi * x[:,0:1])
        return torch.mean((y - target)**2)
    
    def analytic_func(self, x, mu):
        u_vals, cache = burgers_reference_at(x, mu)
        return torch.tensor(u_vals)

    


def _burgers_fd_grid(nu=0.01, nx=401, nt=401):
    x = np.linspace(-1.0, 1.0, nx)
    dx = x[1] - x[0]
    t0, t1 = 0.0, 1.0

    u = -np.sin(np.pi * x)       # 初值
    u[0] = 0.0; u[-1] = 0.0      # Dirichlet 边界

    def flux(u): return 0.5 * u * u

    # CFL 步长
    dt_diff = 0.45 * dx * dx / (2.0 * nu + 1e-12)
    dt_conv = 0.45 * dx / (np.max(np.abs(u)) + 1e-6)
    dt = min(dt_diff, dt_conv)
    steps = int(np.ceil((t1 - t0) / dt))
    dt = (t1 - t0) / steps

    # 时间步推进，保存每一步
    U = np.zeros((steps + 1, x.size), dtype=np.float64)  # [steps+1, nx]
    U[0] = u.copy()
    for k in range(steps):
        uL = u[:-1]; uR = u[1:]
        fL = flux(uL); fR = flux(uR)
        alpha = np.maximum(np.abs(uL), np.abs(uR))
        f_half = 0.5*(fL + fR) - 0.5*alpha*(uR - uL)        # Lax-Friedrichs/Rusanov
        conv = (f_half[1:] - f_half[:-1]) / dx               # (f)_x
        diff = nu * (u[:-2] - 2*u[1:-1] + u[2:]) / (dx*dx)   # u_xx

        u_new = u.copy()
        u_new[1:-1] = u[1:-1] - dt*conv + dt*diff
        u_new[0] = 0.0; u_new[-1] = 0.0
        u = u_new
        U[k+1] = u.copy()

    # 生成整齐的时间轴（steps+1 个节点）
    t = np.linspace(t0, t1, steps + 1)
    return x, t, U.T  # 统一用 [nx, nt] 形状返回


# -------- 双线性插值工具：在规则网格 (x_grid, t_grid) 上插值 U --------
def _bilinear_interp_rect(U, x_grid, t_grid, xq, tq):
    """
    U: [nx, nt] 对应 x_grid × t_grid
    xq, tq: 查询点数组，形状 [N]
    返回: u(xq, tq) [N]
    """
    nx, nt = U.shape
    x_min, x_max = x_grid[0], x_grid[-1]
    t_min, t_max = t_grid[0], t_grid[-1]

    # 裁剪到边界内（防越界）
    xq = np.clip(xq, x_min, x_max)
    tq = np.clip(tq, t_min, t_max)

    # 找到每个查询点所在的左侧索引
    ix = np.searchsorted(x_grid, xq) - 1
    it = np.searchsorted(t_grid, tq) - 1
    ix = np.clip(ix, 0, nx-2)
    it = np.clip(it, 0, nt-2)

    x0 = x_grid[ix];     x1 = x_grid[ix+1]
    t0 = t_grid[it];     t1 = t_grid[it+1]
    # 权重（避免除零）
    wx = np.where(x1 > x0, (xq - x0)/(x1 - x0), 0.0)
    wt = np.where(t1 > t0, (tq - t0)/(t1 - t0), 0.0)

    # 四个栅格点
    f00 = U[ix,     it    ]
    f10 = U[ix + 1, it    ]
    f01 = U[ix,     it + 1]
    f11 = U[ix + 1, it + 1]

    # 双线性插值
    u = (1-wx)*(1-wt)*f00 + wx*(1-wt)*f10 + (1-wx)*wt*f01 + wx*wt*f11
    return u


# def burgers_reference_at(X, T, nu=0.01, grid_nx=801, grid_nt=801, _cache=None):
#     """
#     X, T: 可是标量、1D数组，或同形状的 numpy 数组（单位：x∈[-1,1], t∈[0,1]）
#     nu:   黏性系数
#     grid_nx, grid_nt: 生成参考解的规则网格分辨率（越大越精细但更慢）
#     _cache: 可选的 (x_grid, t_grid, U) 缓存，用于同一个 nu 的多次查询时复用

#     返回：与 X/T 同形状的 u 数组
#     """
#     X_arr = np.asarray(X, dtype=np.float64)
#     T_arr = np.asarray(T, dtype=np.float64)
#     assert X_arr.shape == T_arr.shape, "X 和 T 需要形状一致"

#     if _cache is None:
#         xg, tg, U = _burgers_fd_grid(nu=float(nu), nx=grid_nx, nt=grid_nt)
#     else:
#         xg, tg, U = _cache

#     u_flat = _bilinear_interp_rect(U, xg, tg, X_arr.ravel(), T_arr.ravel())
#     return u_flat.reshape(X_arr.shape), (xg, tg, U)   # 同时返回缓存以便复用


def burgers_reference_at(X, nu=0.01, grid_nx=101, grid_nt=101, _cache=None):
    X_arr = X.detach().cpu().numpy()
    nu = nu.detach().cpu().numpy()
    X_arr = np.asarray(X_arr, dtype=np.float64)
    assert X_arr.shape[-1] == 2

    if _cache is None:
        xg, tg, U = _burgers_fd_grid(nu=float(nu), nx=grid_nx, nt=grid_nt)
    else:
        xg, tg, U = _cache

    xq = X_arr[:, 0]
    tq = X_arr[:, 1]

    u_flat = _bilinear_interp_rect(U, xg, tg, xq, tq)
    return u_flat, (xg, tg, U)


if __name__ == "__main__":
    N = 10
    X = np.zeros((N,2))
    X[:,0] = np.linspace(-0.8, 0.9, N)   # x
    X[:,1] = np.linspace(0.0, 1.0, N)    # t
    nu = 0.05

    u_vals, cache = burgers_reference_at(X, nu=nu, grid_nx=201, grid_nt=201)

    xg, tg, U = cache

    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    # 左：全局 landscape
    im = ax[0].imshow(U.T, origin='lower',
                    extent=[xg[0], xg[-1], tg[0], tg[-1]],
                    aspect='auto')
    ax[0].set_title(f"Burgers solution landscape (nu={nu})")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("t")
    fig.colorbar(im, ax=ax[0])

    # 右：插值点
    ax[1].imshow(U.T, origin='lower',
                extent=[xg[0], xg[-1], tg[0], tg[-1]],
                aspect='auto', alpha=0.6)
    sc = ax[1].scatter(X[:,0], X[:,1], c=u_vals, cmap='coolwarm', edgecolors='k', s=80)
    ax[1].set_title("Queried points with interpolated values")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("t")
    fig.colorbar(sc, ax=ax[1])

    plt.tight_layout()
    plt.savefig('burgers_2d.png')

# def burgers_2d(num_layers, num_nodes, activation_func, epochs, grid_size, learning_rate, test_name, plotshow = False, device ='cpu'):

#     seed = 42
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)

#     data = np.load('Burgers.npz')
#     t, x, usol = data['t'], data['x'], data['usol']
#     len_x = x.shape[0]
#     len_t = t.shape[0]
#     data.close()
#     points_test = np.column_stack((np.meshgrid(x, t, indexing='xy')[0].ravel(), np.meshgrid(x, t, indexing='xy')[1].ravel()))
#     true_test = usol.T.ravel()
#     points_test = torch.from_numpy(points_test).float().to(device)
#     true_test = torch.from_numpy(true_test).float().to(device).unsqueeze(1)

#     # PDEs
#     def pde(x, y):
#         x.requires_grad_(True)
#         dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
#         dy_t = dy[:, 1:2]
#         dy_x = dy[:, 0:1]
#         dy_xx = torch.autograd.grad(dy_x, x, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
#         return dy_t + y * dy_x - 0.01 * dy_xx

#     # def output_transform(x, y):
#     #     return -torch.sin(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y

#     def plot_results(model, points_test, true_test, x, t, test_name):
#         model.eval()
#         len_x = len(x)
#         len_t = len(t)

#         with torch.no_grad():
#             pred_test = model(points_test).detach().cpu().numpy()

#         pred_field = pred_test.reshape(len_t, len_x).T  # 形状为(空间点数, 时间点数)
#         true_field = true_test.cpu().numpy().reshape(len_t, len_x).T
#         error_field = np.abs(pred_field - true_field)

#         X, T = np.meshgrid(x, t, indexing='ij')  # 空间x作为行，时间t作为列

#         plt.figure(figsize=(5, 4))
#         plt.pcolormesh(X, T, pred_field, cmap='viridis', shading='gouraud')
#         plt.title('Predicted Field', fontsize=15)
#         plt.colorbar()
#         plt.xlabel('x', fontsize=15)
#         plt.ylabel('t', fontsize=15)
#         plt.tight_layout()
#         # plt.savefig(f'predicted_field_{test_name}.png', dpi=300)
#         plt.show()

#         plt.figure(figsize=(5, 4))
#         plt.pcolormesh(X, T, true_field, cmap='viridis', shading='gouraud')
#         plt.title('Ground Truth', fontsize=15)
#         plt.colorbar()
#         plt.xlabel('x', fontsize=15)
#         plt.ylabel('t', fontsize=15)
#         plt.tight_layout()
#         # plt.savefig(f'true_field_{test_name}.png', dpi=300)
#         plt.show()

#         plt.figure(figsize=(5, 4))
#         plt.pcolormesh(X, T, error_field, cmap='hot_r', shading='gouraud')
#         plt.title('Absolute Error', fontsize=15)
#         plt.colorbar()
#         plt.xlabel('x', fontsize=15)
#         plt.ylabel('t', fontsize=15)
#         plt.tight_layout()
#         # plt.savefig(f'error_field_{test_name}.png', dpi=300)
#         plt.show()

#     # Loss function
#     def losses(model, points, b1, b2, b3, b4, points_test, lam_pde=1.0, lam_bc=1.0, lam_ic=1.0):
#         points = points.clone().requires_grad_(True)
#         pred_points = model(points)
#         pde_residual = pde(points, pred_points)
#         pde_loss = torch.mean(pde_residual ** 2)

#         pred_b1 = model(b1)
#         pred_b2 = model(b2)
#         bc_loss = torch.mean(pred_b1 ** 2) + torch.mean(pred_b2 ** 2)

#         pred_b3 = model(b3)
#         true_b3 = -torch.sin(np.pi * b3[:, 0:1])
#         ic_loss = torch.mean((pred_b3 - true_b3) ** 2)

#         pred_test = model(points_test)
#         l1_ab_metric = torch.mean(torch.abs(pred_test - true_test))
#         l1_re_metric = torch.mean(torch.abs(pred_test - true_test)) / torch.mean(torch.abs(true_test))
#         l2_ab_metric = torch.mean((pred_test - true_test) ** 2)
#         l2_re_metric = torch.sqrt(torch.mean((pred_test - true_test) ** 2)) / torch.sqrt(torch.mean(true_test ** 2))

#         total_loss = lam_pde * pde_loss + lam_bc * bc_loss + lam_ic * ic_loss

#         return total_loss, {
#             'pde_loss': pde_loss.item(),
#             'bc_loss': bc_loss.item(),
#             'ic_loss': ic_loss.item(),
#             'total_loss': total_loss.item(),
#             'l1_ab_metric': l1_ab_metric.item(),
#             'l1_re_metric': l1_re_metric.item(),
#             'l2_ab_metric': l2_ab_metric.item(),
#             'l2_re_metric': l2_re_metric.item()
#         }

#     # Initialization
#     def init_weights(m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight)
#             m.bias.data.fill_(0.01)

#     try:
#         start_time = time.time()

#         # Generate training and test points
#         num_x = grid_size
#         num_y = grid_size
#         range_x = torch.tensor([[-1., 1], [0., 0.99]]).to(device)

#         points, b1, b2, b3, b4 = get_data_2d(num_x, num_y, range_x, device)

#         # Create model with specified parameters
#         model = PINNs(in_dim=2, hidden_dim=num_nodes, out_dim=1, num_layer=num_layers, activation=activation_func).to(device)
#         # model.apply_output_transform(output_transform)
#         model.apply(init_weights)

#         # Setup optimizer
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         # optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')

#         # Training loop
#         train_losses_history = []

#         for epoch in tqdm(range(epochs), desc=f"Training {test_name}"):
#             def closure():
#                 total_loss, train_loss_dict = losses(model, points, b1, b2, b3, b4, points_test, lam_pde=1.0, lam_bc=1.0, lam_ic=1.0)
#                 optimizer.zero_grad()
#                 total_loss.backward()
#                 if not hasattr(closure, 'latest_loss_dict'):
#                     closure.latest_loss_dict = train_loss_dict
#                 else:
#                     closure.latest_loss_dict.update(train_loss_dict)
#                 return total_loss

#             optimizer.step(closure)

#             if hasattr(closure, 'latest_loss_dict'):
#                 train_losses_history.append(closure.latest_loss_dict.copy())

#         end_time = time.time()
#         runtime = end_time - start_time
#         if plotshow == True:
#             plot_results(model, points_test, true_test, x, t, test_name)

#         # Get final L2 error
#         final_l1_ab_error = train_losses_history[-1]['l1_ab_metric']
#         final_l1_re_error = train_losses_history[-1]['l1_re_metric']
#         final_l2_ab_error = train_losses_history[-1]['l2_ab_metric']
#         final_l2_re_error = train_losses_history[-1]['l2_re_metric']
#         final_pde_loss = train_losses_history[-1]['pde_loss']
#         final_bc_loss = train_losses_history[-1]['bc_loss']
#         final_ic_loss = train_losses_history[-1]['ic_loss']
#         final_total_loss = train_losses_history[-1]['total_loss']


#         print(f"\n{test_name} completed:")
#         print(f"Runtime: {runtime:.2f} seconds")
#         print(f"Final L2 error: {final_l2_ab_error:.4e}")
#         print(f"Configuration: layers={num_layers}, nodes={num_nodes}, activation={activation_func}, epochs={epochs}, sample_size={grid_size}, lr={learning_rate}")

#         return {
#             'test_name': test_name,
#             'final_l1_ab_error': final_l1_ab_error,
#             'final_l1_re_error': final_l1_re_error,
#             'final_l2_ab_error': final_l2_ab_error,
#             'final_l2_re_error': final_l2_re_error,
#             'final_pde_loss': final_pde_loss,
#             'final_bc_loss': final_bc_loss,
#             'final_ic_loss': final_ic_loss,
#             'final_total_loss': final_total_loss,
#             'runtime': runtime,
#             'num_layers': num_layers,
#             'num_nodes': num_nodes,
#             'activation_func': activation_func,
#             'epochs': epochs,
#             'grid_size': grid_size,
#             'learning_rate': learning_rate
#         }

#     except Exception as e:
#         print(f"Error in {test_name}: {str(e)}")
#         return {
#             'test_name': test_name,
#             'final_l2_error': float('inf'),
#             'runtime': 0,
#             'error': str(e)
#         }