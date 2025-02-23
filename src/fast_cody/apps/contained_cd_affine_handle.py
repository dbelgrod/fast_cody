import os
import numpy as np
import scipy as sp
import igl
import json
from os.path import basename, splitext

import fast_cody as fc

def contained_cd_affine_handle(msh_file=None, texture_png=None, texture_obj=None):
    """
    Runs a standard interactive fast CD simulation, where the user can manipulate a single affine
    handle with a Guizmo and observe secondary effects in real-time.
    """
    if msh_file is not None:
        [V, F, T] = fc.read_msh(msh_file)
    elif msh_file is None:
        msh_file = fc.get_data("./cd_fish.msh")
        [V, F, T] = fc.read_msh(msh_file)

    if texture_png is None or texture_obj is None:
        if msh_file ==  fc.get_data("./cd_fish.msh"):
            texture_png = fc.get_data("./cd_fish_tex.png")
            texture_obj = fc.get_data("./cd_fish_tex.obj")

    #TODO: Scale and center geometry
    # const Eigen::MatrixXd& V, const  double h, const Eigen::RowVector3d c , double& so, Eigen::RowVector3d& to)
    def scale_and_center_geometry(V, h=1, c=np.array([0, 0, 0])):
        so = h / (np.max(V[:, 1]) - np.min(V[:, 1]))
        V2 = V * so
        to = np.mean(V2, axis=0)
        V2 = V2 - to
        return V2, so, to
    
    V_mod, so, to = scale_and_center_geometry(V)
    
    #TODO: Linear Blend Skinning Jacobian
    Wp = np.ones((V.shape[0], 1)) #single handle skinning weight
    
    def lbs_jacobian(V, Wp):
        # Check dimensions
        assert V.shape[0] == Wp.shape[0], "Weights should have same number of rows as V!"
        assert V.shape[1] == 3, "Can only handle 3D case for now"

        n = V.shape[0]
        dim = V.shape[1]
        b = Wp.shape[1]
        
        ones_n = np.ones((n, 1))
        U = np.hstack((V, ones_n))

        ones_b = np.ones((1, b))  # row vector of b ones
        U_exp = np.kron(ones_b, U)

        ones_dp1 = np.ones((1, dim + 1))  # row vector of d+1 ones
        W_exp = np.kron(Wp, ones_dp1)

        J_compact = U_exp * W_exp  # element-wise product

        # Repeat J_compact along diagonal dim times
        m, n = J_compact.shape
        J = np.zeros((m*dim, n*dim))
        for k in range(dim):
            J[k*m:(k+1)*m, k*n:(k+1)*n] = J_compact
        
        return J
    
    J = lbs_jacobian(V_mod, Wp)
    
    #TODO: Complementary Constraint Matrix
    C = fc.complementary_constraint_matrix(V_mod, T, J, dt=1e-3)
    
    #TODO: LBS Weight Space Constraint
    C2 = fc.lbs_weight_space_constraint(V_mod, C)
    
    #TODO: Skinning Subspace
    num_modes = 16
    num_clusters = 100
    constraint_enforcement = "optimal"
    read_cache = False
    cache_dir = None
    [B, l, Ws] = fc.skinning_subspace(V_mod, T, num_modes, num_clusters, C=C2, read_cache=read_cache,
                                         cache_dir=cache_dir, constraint_enforcement=constraint_enforcement)
    
    #TODO: Fast CD Sim
    sim = fc.fast_cd_sim(V_mod, T, B, l, J, mu=1e4, rho=1e3, h=1e-2, cache_dir=cache_dir, read_cache=read_cache)
    
    #TODO: Set sim state and initial rig parameters
    z0 = np.zeros((num_modes*12, 1))
    T0 = np.identity(4).astype( dtype=np.float32, order="F");
    p0 = T0[0:3, :].reshape((12, 1))
    st = fc.fast_cd_state(z0, p0)
    
    #TODO: Pre-draw callback
    step = 0
    def pre_draw_callback():
        return
        nonlocal J, B, T0, sim, st, step
        p = viewer.T0[0:3, :].reshape( (12, 1))
        z = sim.step( p, st)
        st.update(z, p)
        # U = np.reshape(J @ p + B @ z, (J.shape[0]//3, 3), order="F") # full positions
        viewer.update_subspace_coefficients(z, p)
        step += 1
    