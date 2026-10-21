import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def calculate_8d_metrics(model, z_query, steps=25):
    """
    Extracts 8 Topological ODE Features for Synthetic Image Detection.
    """
    model.eval()
    dt = 1.0 / steps 
    batch_size = z_query.shape[0]

    z_query = z_query.to(device).float()
    
    # ODE INTEGRATION
    z_t = z_query.clone()
    u_list = []
    
    for i in range(steps):
        t_val = 1.0 - (i * dt)
        t_tensor = torch.full((batch_size, 1), t_val, device=device)
        
        u_i = model(z_t, t_tensor)
        u_list.append(u_i)
        z_t = z_t - dt * u_i
        
    z_0 = z_t.clone()
    
    # BASE TENSOR PREPARATION
    V = torch.stack(u_list, dim=1) 
    v_norms = torch.norm(V, p=2, dim=-1)
    
    cos_sims = F.cosine_similarity(V[:, :-1, :], V[:, 1:, :], dim=-1)
    k_t = 1.0 - cos_sims
    
    # FEATURE EXTRACTION
    # 1. Mean Angular Curvature
    k_cos = torch.mean(k_t, dim=1)
    
    # 2. Path Length
    path_length = torch.sum(v_norms * dt, dim=1)
    
    # 3. Transport Efficiency
    endpoint_dist = torch.norm(z_query - z_0, p=2, dim=-1)
    actual_energy = torch.sum(v_norms**2 * dt, dim=1)
    r_eff = actual_energy / (endpoint_dist**2 + 1e-8)
    
    # 4. Velocity Variance
    vel_variance = torch.var(v_norms, dim=1, unbiased=False)
    
    # 5. Curvature Variance
    curv_variance = torch.var(k_t, dim=1, unbiased=False)
    
    # 6. Maximum Curvature
    max_curv = torch.max(k_t, dim=1)[0]
    
    # 7. Trajectory Entropy
    V_dir = F.normalize(V, p=2, dim=-1)
    Gram = torch.bmm(V_dir, V_dir.transpose(1, 2)) / steps
    eigenvalues = torch.linalg.eigvalsh(Gram)
    eigenvalues = torch.clamp(eigenvalues, min=0.0) 
    p = eigenvalues / (torch.sum(eigenvalues, dim=-1, keepdim=True) + 1e-8)
    traj_entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1)
    
    # 8. Normalized Endpoint Error
    norm_endpoint_error = endpoint_dist / (path_length + 1e-8)
    
    return (
        k_cos.cpu().numpy(),
        r_eff.cpu().numpy(),
        path_length.cpu().numpy(),
        vel_variance.cpu().numpy(),
        curv_variance.cpu().numpy(),
        max_curv.cpu().numpy(),
        traj_entropy.cpu().numpy(),
        norm_endpoint_error.cpu().numpy()
    )