import os
import numpy as np
import scipy as sp
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import LinearOperator
from sklearn.cluster import KMeans

import igl
import json
from os.path import basename, splitext
import cvxopt
import cvxopt.umfpack
import time

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
    
    V, so, to = scale_and_center_geometry(V)
    
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
    
    J = lbs_jacobian(V, Wp)
    
    #TODO: Complementary Constraint Matrix
    
    def laplacian(X, T, mu=None):
        """
        Computes the Laplacian of a d-simplex. With d being 2, 3 or 4.

        Parameters
        ----------
        V : (n, 3) numpy float array
            Vertex positions
        T : (t, d)  (d=2, 3, 4) numpy int array
            Simplex indices
        mu : (t,) numpy float array
            Per-simplex conductivity

        Returns
        -------
        L : (n, n) scipy sparse float array
            Laplacian matrix
        """
        if mu is None:
            mu = np.ones(T.shape[0])
        elif np.isscalar(mu):
            mu = mu * np.ones(T.shape[0])
        else:
            assert (type(mu) == np.ndarray)
            assert (mu.shape[0] == T.shape[0])

        if T.shape[1] == 2:
            l = igl.edge_lengths(X, T)
            I = np.hstack((T[:, 0], T[:, 1], T[:, 0], T[:, 1]))
            J = np.hstack((T[:, 1], T[:, 0], T[:, 0], T[:, 1]))
            VV = np.hstack((-1.0 / l * mu, -1.0 / l * mu, 1.0 / l * mu, 1.0 / l * mu))
            L = sp.sparse.csc_matrix((VV, (I, J)), shape=(X.shape[0], X.shape[0]))
            return L
        if T.shape[1] == 3 or T.shape[1] == 4:
            muv = np.kron(mu, np.ones(3))
            Muv = sp.sparse.diags(muv)
            J = igl.grad(X, T)
            a = igl.volume(X, T)
            A = sp.sparse.kron(sp.sparse.identity(3), sp.sparse.diags(a))
            L = J.T @ A @ Muv @ J
            L = L[:X.shape[0], :][:, :X.shape[0]]
            return L
        else:
            raise NotImplementedError("Laplacian not implemented for dimension %d" % T.shape[1])

    def umfpack_lu_solve(A, b):
        """
        Solves Ax = b using LU factorization with umfpack.
        Parameters
        ----------
        A : (n, n) float numpy array
            Matrix to solve
        b : (n, ) float numpy array
            Right hand side

        Returns
        -------
        x : (n, ) float numpy array
            Solution to Ax = b
        """
        [I, J] = A.nonzero()
        v = A.data
        Ac= cvxopt.spmatrix(v, I, J, A.shape)
        bc = cvxopt.matrix(b)
        cvxopt.umfpack.linsolve(Ac, bc)
        cnp = np.array(bc)
        return cnp

    def diffuse_weights(Vv, Tv, phi, bI,  dt=None, normalize=True):
        """ Performs a diffusion on the tet mesh Vv, Tv at nodes bI for time dt.

        Parameters
        ----------
        Vv : (n, 3) float numpy array
            Mesh vertices
        Tv : (t, 4) int numpy array
            Mesh tets
        phi : (c, b) float numpy array
            Quantity to diffuse
        bI : (c, b) int numpy array
            Indices at diffusion points
        dt : float
            Time to diffuse for
        normalize : bool
            Whether to normalize the weights

        Returns
        -------
        W : (n, b) float numpy array
            Diffused quantities over entire mesh

        """

        if (dt is None):
            dt = np.mean(igl.edge_lengths(Vv, Tv)) ** 2

        L = laplacian(Vv, Tv)
        M = igl.massmatrix(Vv, Tv)

        Q = L * dt + M

        ii = np.setdiff1d(  np.arange(Q.shape[0]), bI)
        # selection matrix for indices bI
        Qii = Q[ii, :][:, ii]
        Qib = Q[ii, :][:, bI]

        Wii = umfpack_lu_solve(Qii, -Qib @ phi)
        W = np.zeros((L.shape[0], Wii.shape[1]))
        W[ii, :] = Wii
        W[bI, :] = phi
        # normalize between 0 and 1
        if W.ndim == 1:
            W = W[:, None]
        if normalize:
            W = (W - np.min(W, axis=0)) / (np.max(W, axis=0) - np.min(W, axis=0))

        return W

    def momentum_leaking_matrix(V, T, dt=None, pow=1):
        # Compute cotangent Laplacian and mass matrices
        F = igl.boundary_facets(T)
        M = igl.massmatrix(V, T)
        Me = sp.sparse.kron(sp.sparse.identity(3), M)
        bI = np.unique(F)
        phi = np.ones((bI.shape[0], 1))
        d = 1 - np.power(diffuse_weights(V, T, phi, bI, dt=dt), pow)

        D = sp.sparse.kron(sp.sparse.identity(3), sp.sparse.diags(d[:, 0]))
        return D

    
    def complementary_constraint_matrix(V, T, J, dt=1e-3):
        M = igl.massmatrix(V, T)
        Me = sp.sparse.kron(sp.sparse.identity(3), M)
        D = momentum_leaking_matrix(V, T, dt=dt)
        C =  (Me @ D @ J).T
        return C
    
    C = complementary_constraint_matrix(V, T, J, dt=1e-3)

    #TODO: LBS Weight Space Constraint
    def orthonormalize(B, M=None):
        """
        Orthonormalize a matrix B with respect to the mass matrix M by cutting off redundant columns.

        Parameters
        ----------
        A : (n, d) float numpy array
            Matrix to orthonormalize
        M : (n, n) scipy sparse matrix
            Mass matrix

        Returns
        -------
        B : (n, d') float numpy array
            Orthonormalized matrix satisfying B.T @ M @ B = I
        """
        if M is None:
            M = sp.sparse.identity(B.shape[0])
        # M = sp.sparse.identity(B.shape[0])

        msqrt = np.sqrt(M.diagonal())
        msqrti = 1 / msqrt
        Msqrt = sp.sparse.diags(msqrt, 0)
        Msqrti = sp.sparse.diags(msqrti, 0)
        Bm = Msqrt @ B


        [Q, R] = np.linalg.qr(Bm, mode='reduced')

        [U, s, V] = np.linalg.svd(Bm, full_matrices=False)

        sI = np.where(s > 1e-14)[0]
        S = np.diag(s)
        B2 = U @ S[:, sI] @ V[sI, :][:, sI]

        B3 = Msqrti @ B2

        return B3
    
    def lbs_weight_space_constraint(V, C):
        """ Rewrites a linear equality constraint that acts on per-vertex displacements (CU(W) = 0)
            to instead act on the per-vertex skinning weights  (AW = 0).

        Parameters
        ----------
        V : (n, d) float numpy array
            Mesh vertices
        C : (c, dn) float numpy array
            Linear equality constraint matrix that acts on per-vertex displacements

        Returns
        -------
        A : (n, c') float numpy array
            Linear equality constraint matrix that acts on per-vertex skinning weights
        """
        C = C.T
        n = V.shape[0]
        d = V.shape[1]

        v = np.ones((n, 1))

        A = np.zeros((0, n))
        for i in range(0, d):
            Id = np.arange(0, n) + i * n
            Jd = np.arange(0, n)
            Pd = sp.sparse.coo_matrix((v.flatten(), (Id, Jd)), shape=(3*n,n))

            for j in range(0, d):
                Vj = V[:, j]
                Adj = C.T @ Pd @ sp.sparse.diags(Vj, 0)
                A = np.vstack([A, Adj])
            Ad1 = C.T @ Pd
            A = np.vstack([A, Ad1])

        W = A
        W2 = orthonormalize(W.T).T
        return W2
    
    C2 = lbs_weight_space_constraint(V, C)
    
    #TODO: Skinning Subspace
    num_modes = 16
    num_clusters = 100
    constraint_enforcement = "optimal"
    read_cache = False
    cache_dir = "./cache/"
    
    def umfpack_lu_solve(A, b):
        """
        Solves Ax = b using LU factorization with umfpack.
        Parameters
        ----------
        A : (n, n) float numpy array
            Matrix to solve
        b : (n, ) float numpy array
            Right hand side

        Returns
        -------
        x : (n, ) float numpy array
            Solution to Ax = b
        """
        [I, J] = A.nonzero()
        v = A.data
        Ac= cvxopt.spmatrix(v, I, J, A.shape)
        bc = cvxopt.matrix(b)
        cvxopt.umfpack.linsolve(Ac, bc)
        cnp = np.array(bc)
        return cnp
    
    class umfpack_LU_LinearOperator(LinearOperator):
        def __init__(self, A):
            self.A = A
            self.shape = A.shape
            self.dtype = A.dtype
            self.A = A

            [I, J] = A.nonzero()
            v = A.data
            Ac = cvxopt.spmatrix(v, I, J, A.shape)
            F = cvxopt.umfpack.symbolic(Ac)
            self.numeric = cvxopt.umfpack.numeric(Ac, F)
            self.A = Ac
            # bc = cvxopt.matrix(b)
            # cvxopt.umfpack.linsolve(Ac, bc)
            # cnp = np.array(bc)
            super(umfpack_LU_LinearOperator, self).__init__( A.dtype ,A.shape)

        def _matvec(self, v):
            b = cvxopt.matrix(v)
            x = b
            cvxopt.umfpack.solve(self.A, self.numeric, b)
            return x

    '''
    Computes Generalized Eigenvalues and Eigenvectors of sparse non-definite matrix A, with massmatrix M

    Inputs:
    A - n x n indefinite sparse matrix

    Optional
    k - int number of eigenvectors/values tos olve for (default=5)
    M - n x n indefinite mass matrix

    Returns
    D - k x 1 eigenvalues
    B - n x k eigenvectors
    '''
    def eigs(A, k=5, M=None):
        """
        Computes Generalized Eigenvalues and Eigenvectors of sparse non-definite matrix A, with massmatrix M

        Parameters
        ----------
        A : (n, n) float sparse matrix
            Indefinite sparse matrix
        k : int
            Number of eigenvectors/values to solve for (default=5)
        M : (n, n) float sparse matrix
            Indefinite mass matrix

        Returns
        --------
        D : (k, 1) float numpy array
            Eigenvalues
        B : (n, k) float numpy array
            Eigenvectors

        """
        if M is None:
            M = sp.sparse.identity(A.shape[0])

        try:
            OpInv = umfpack_LU_LinearOperator(A)
            # MInv = umfpack_LU_LinearOperator(M)
            [D, B] = sp.sparse.linalg.eigs(A, M=M, k=k, sigma=0,
                                    which='LM', OPinv=OpInv)
        except:
            print("UMFPACK LU Factorization Failed, Trying Scipy LU, which is slower")
            [D, B] = sp.sparse.linalg.eigs(A, M=M, k=k, sigma=0,which='LM')
        return D, B
    
    def laplacian_eigenmodes(V, T, m, read_cache=False, cache_dir=None, J=None,
                         mu=None, constraint_enforcement="optimal"):
        """ Computes Laplacian Eigenmodes for a given mesh.

        Parameters
        ----------
        V : (n, 3) float numpy array
            Vertex positions
        T : (F, 4) int numpy array
            Tet indices
        m : int
            Number of modes to compute
        read_cache : bool
            Whether to read from cache or not (default False)
        cache_dir : str
            Directory to cache results in (default None)
        J : (c x n) float numpy array
            Constraint matrix we desire on our weights s.t. J @ W = 0 (default None)
        mu : float
            Per-tet conducivity. If None, sets it to 1.0 everyewhere (default None)
        constraint_enforcement : str
            Method of enforcing constraint. Either "project" or "optimal" (default "optimal")

        Returns
        -------
        B : (n, m) float numpy array
            Subspace matrix/Eigenvectors of laplacian.
        E : (m, 1) float numpy array
            Eigenvalues of each eigenvector
        """
      
        L = laplacian(V, T, mu=mu)
        M = igl.massmatrix(V, T)
        L =  L + 1e-8 * M
        if constraint_enforcement == "optimal":
            if J is not None:
                c = J.shape[0]
                Z = sp.sparse.csc_matrix((c, c))
                L = vstack((hstack((L, J.T)), hstack((J, Z )))).tocsc()
                M = sp.sparse.block_diag((M, Z)).tocsc()
        print("Computing eigenmodes... may take a while...")
        start = time.time()
        [E, B] = eigs(L, M=M, k=m)
        print("Done computing eigenmodes! Took, ", time.time() - start, " seconds")

        n = V.shape[0]
        if J is not None:
            B = B[:n, :]

        B = np.real(B)
        E = np.real(E)

        # if constraint_enforcement == "project":
        #     B = project_out_subspace(B, J.T)
        #     E = np.diag(B.T @ L @ B)
        #     print("Done projecting out constraints from eigenmodes")


        B = orthonormalize(B)

        return B, E
    
    def average_onto_simplex(A, T):
        """ Average quantity from vertices to simplices
        Parameters
        ----------
        A : (n, d) numpy float array
            Per vertex d-dimensional quantities
        T : (t, s) numpy int array
            Simplex indices

        Returns
        -------
        At : (t, d) numpy float array
            Per simplex d-dimensional quantities
        """
        At = np.zeros((T.shape[0], A.shape[1]))
        for td in range(T.shape[1]):
            At += (A[T[:, td], :])/T.shape[1]

        return At
        
    def skinning_clusters(W, D, T, k, l=2, num_clustering_features=10,
                      return_centroids=False, return_simplex_features=False):
        """ Skinning clusters.

        Parameters
        ----------
        W : numpy float array
            n x b skinning weights
        D : numpy float array
            b x 1 eigenvalue/weighing given to each skinning weight
        T : numpy int array
            T x 4 tet geometry
        k : int
            number of clusters
        l : int
            power to raise D to
        num_clustering_features : int
            number of features to use for clustering
        return_centroids : bool
            whether to return the centroids of the clusters
        return_simplex_features : bool
            whether to return the features averaged over each tet
        """
        num_clustering_features = min(num_clustering_features, W.shape[1])
        # need to average the skinning weights over each tet
        assert(T.shape[1] == 4, "only tets implemented so far for clustering")

        Wt = average_onto_simplex(W, T)
        # Wt2 = Wt / np.power(D, 2)
        Wt = Wt / np.power(D, l)
        Wt = Wt[:, 0:num_clustering_features]
        kmeans = KMeans(n_clusters=k, random_state=0).fit(Wt)
        l = kmeans.labels_

        if return_simplex_features:
            return l, Wt
        if return_centroids == True:
            return l, kmeans.cluster_centers_
        else:
            return l
    
    def skinning_subspace(X, T, num_modes, num_clusters,
                        ortho=True, mu=None, C=None, constraint_enforcement="optimal"):
        """
        Constructs a physics subspace corresponding with skinning eigenmodes and skinning clusters

        Parameters
        ----------
        X : (n, 3) float numpy array
            Vertex positions
        T : (t, 4) int numpy array
            Tet indices
        num_modes : int
            Number of modes to use
        num_clusters : int
            Number of clusters to use
        ortho : bool
            Whether to orthonormalize the subspace
        mu : float numpy array
            Per-tet conducivity. If None, sets it to 1.0 everyewhere
        C : (3n, c) float numpy array
            Constraint matrix we desire on our weights s.t. C.T @ W = 0
        constraint_enforcement : str
            Method of enforcing constraint. Either "project" or "optimal"

        Returns
        -------
        B : (3n, m) float numpy array
            Subspace matrix/Eigenvectors of laplacian.
        l : (t, 1) int numpy array
            Cluster indices
        W : (n, m) float numpy array
            Skinning weights


        """
        dim = X.shape[1]
        [W, E] = laplacian_eigenmodes(X, T, num_modes, read_cache=False, mu=mu, J=C, constraint_enforcement=constraint_enforcement)
        B = lbs_jacobian(X, W)
        M = sp.sparse.kron(sp.sparse.identity(3), igl.massmatrix(X, T))
        l = skinning_clusters(W, E, T, num_clusters, l=2, num_clustering_features=num_modes)

        return B, l, W
    
    [B, l, Ws] = skinning_subspace(V, T, num_modes, num_clusters, C=C2, constraint_enforcement=constraint_enforcement)
    
    #TODO: Fast CD Sim
    mu = 1e4
    rho = 1e3
    h = 1e-2
    sim = fc.fast_cd_sim(V, T, B, l, J, mu=mu, rho=rho, h=h, cache_dir=cache_dir, read_cache=read_cache)
    
    #TODO: Set sim state and initial rig parameters
    z0 = np.zeros((num_modes*12, 1))
    T0 = np.identity(4).astype( dtype=np.float32, order="F");
    p0 = T0[0:3, :].reshape((12, 1))
    st = fc.fast_cd_state(z0, p0)
    
    #TODO: Pre-draw callback
    step = 0
    def pre_draw_callback():
        nonlocal J, B, T0, sim, st, step
        p = viewer.T0[0:3, :].reshape( (12, 1))
        z = sim.step( p, st)
        st.update(z, p)
        # U = np.reshape(J @ p + B @ z, (J.shape[0]//3, 3), order="F") # full positions
        viewer.update_subspace_coefficients(z, p)
        step += 1
    viewer = fc.viewers.interactive_handle_subspace_viewer(V, T, Wp, Ws,  pre_draw_callback,T0=T0,
                                                  texture_png=texture_png, texture_obj=texture_obj,
                                                  t0=to, s0=so, init_guizmo=True)
    viewer.launch()
