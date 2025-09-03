# %% Imports

import math
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")


# %% Functions
# --- Quadrature: Dunavant degree-5 (7-point) on the reference triangle. ---
# The weights below are scaled so that sum(w) = area(reference) = 1/2.
def _dunavant_rule_deg5():
    xi, eta, w = [], [], []
    # centroid
    xi.append(1 / 3)
    eta.append(1 / 3)
    w.append(0.225000000000000)
    # group 2
    a, b, w2 = 0.059715871789770, 0.470142064105115, 0.132394152788506
    for alpha, beta, gamma in [(a, b, b), (b, b, a), (b, a, b)]:
        xi.append(beta)
        eta.append(gamma)
        w.append(w2)
    # group 3
    a, b, w3 = 0.797426985353087, 0.101286507323456, 0.125939180544827
    for alpha, beta, gamma in [(a, b, b), (b, b, a), (b, a, b)]:
        xi.append(beta)
        eta.append(gamma)
        w.append(w3)
    w = 0.5 * np.array(w)  # scale to half-area of reference triangle
    return np.array(xi), np.array(eta), w


def _triangle_geometry(V):
    e1 = V[1] - V[0]
    e2 = V[2] - V[0]
    n = np.cross(e1, e2)
    J = np.linalg.norm(n)  # |e1 x e2|  (twice area)
    if J == 0.0:
        raise ValueError("Degenerate triangle.")
    n_hat = n / J
    return e1, e2, n_hat, J


def _subdivide_triangle(V):
    m01 = 0.5 * (V[0] + V[1])
    m12 = 0.5 * (V[1] + V[2])
    m20 = 0.5 * (V[2] + V[0])
    return np.array(
        [
            [V[0], m01, m20],
            [m01, V[1], m12],
            [m20, m12, V[2]],
            [m01, m12, m20],
        ]
    )


def kelvin_triangle_constant_traction(
    X,
    V,
    traction,
    mu,
    nu,
    *,
    quad_rule="dunavant5",
    n_subdiv=1,
    backend="auto",
    return_sigma=True,
    fd_eps=1e-6,
):
    """
    Displacement u(x) and (optionally) Cauchy stress sigma(x) from a constant traction
    applied over a flat triangle in an infinite isotropic elastic medium (Kelvin SL).

    Parameters
    ----------
    X : (3,) or (N,3) array_like
        Field point(s).
    V : (3,3) array_like
        Triangle vertices [v0, v1, v2].
    traction : (3,) array_like
        Constant traction vector t (force/area).
    mu : float
        Shear modulus.
    nu : float
        Poisson's ratio (|nu| < 0.5).
    quad_rule : str
        Only 'dunavant5' is provided (good and compact); increase n_subdiv near the facet.
    n_subdiv : int
        Uniform 4-way recursive subdivision level (0–3 typical).
    backend : {'auto','jax','numpy'}
        'jax' uses JAX + autodiff for stresses; otherwise NumPy + finite differences.
    return_sigma : bool
        If True, also returns sigma.
    fd_eps : float
        Finite-difference step for NumPy stress (ignored with JAX).

    Returns
    -------
    u : (N,3) ndarray
    sigma : (N,3,3) ndarray or None
    """
    # -- backend detection --
    has_jax = False
    if backend in ("auto", "jax"):
        try:
            import jax  # type: ignore
            import jax.numpy as jnp
            from jax import jacfwd, vmap  # type: ignore

            has_jax = True
        except Exception:
            if backend == "jax":
                raise RuntimeError("backend='jax' requested but JAX is unavailable.")

    X = np.atleast_2d(np.asarray(X, dtype=float))
    V = np.asarray(V, dtype=float).reshape(3, 3)
    t = np.asarray(traction, dtype=float).reshape(3)

    # material constants
    lam = 2 * mu * nu / (1 - 2 * nu)
    C1 = (3 - 4 * nu) / (16 * math.pi * mu * (1 - nu))
    C2 = 1.0 / (16 * math.pi * mu * (1 - nu))

    # quadrature (degree-5)
    xi, eta, w = _dunavant_rule_deg5()

    # optional uniform subdivision
    triangles = [V]
    for _ in range(max(0, int(n_subdiv))):
        triangles = [t_ for tri in triangles for t_ in _subdivide_triangle(tri)]

    # precompute geometry per subtriangle
    tris_geom = []
    for tri in triangles:
        e1, e2, n_hat, J = _triangle_geometry(tri)
        tris_geom.append((tri, e1, e2, J))

    # ----- numeric integrator for S0 and Q -----
    def _integrate_S0_Q_np(x):
        # S0 = ∫ 1/r dS,   Q = ∫ (r ⊗ r)/r^3 dS, with r = x - y
        S0 = 0.0
        Q = np.zeros((3, 3), dtype=float)
        for tri, e1, e2, J in tris_geom:
            Y = tri[0] + np.outer(xi, e1) + np.outer(eta, e2)  # (nq,3)
            R = x[None, :] - Y  # (nq,3)
            r = np.linalg.norm(R, axis=1)  # (nq,)
            invr = 1.0 / r
            invr3 = invr**3
            fac = w * J
            S0 += np.sum(fac * invr)
            Q += np.einsum("q,qi,qj->ij", fac * invr3, R, R)
        return S0, Q

    # displacements
    U = np.zeros((X.shape[0], 3), dtype=float)
    for i in range(X.shape[0]):
        S0, Q = _integrate_S0_Q_np(X[i])
        U[i] = C1 * S0 * t + C2 * (Q @ t)

    if not return_sigma:
        return U, None

    # stresses
    if has_jax:
        # JAX path: differentiate u(x) exactly (autodiff) for epsilon and sigma
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)  # recommend 64-bit for near-field
        from jax import jacfwd, vmap

        # capture constants into jax arrays
        t_j = jnp.array(t)
        w_j = jnp.array(w)
        C1_j = jnp.array(C1)
        C2_j = jnp.array(C2)
        Js = jnp.array([g[3] for g in tris_geom])  # (M,)
        tri0 = jnp.array([g[0][0] for g in tris_geom])  # (M,3) first vertex
        e1s = jnp.array([g[1] for g in tris_geom])  # (M,3)
        e2s = jnp.array([g[2] for g in tris_geom])  # (M,3)
        xi_j = jnp.array(xi)
        eta_j = jnp.array(eta)

        def u_single(x):
            Y = (
                tri0[:, None, :]
                + xi_j[:, None] * e1s[:, None, :]
                + eta_j[:, None] * e2s[:, None, :]
            )  # (M,nq,3)
            R = x[None, None, :] - Y  # (M,nq,3)
            r = jnp.linalg.norm(R, axis=2)  # (M,nq)
            invr = 1.0 / r
            invr3 = invr**3
            fac = w_j[None, :] * Js[:, None]  # (M,nq)
            S0 = jnp.sum(fac * invr)
            Q = jnp.einsum("mq,mqi,mqj->ij", fac * invr3, R, R)
            return C1_j * S0 * t_j + C2_j * (Q @ t_j)

        J_u = jacfwd(u_single)  # du_i/dx_k

        def sigma_single(x):
            G = J_u(x)  # (3,3)
            eps = 0.5 * (G + G.T)
            tr_eps = jnp.trace(eps)
            lam_j = 2 * mu * nu / (1 - 2 * nu)  # scalar, cast as needed
            return lam_j * tr_eps * jnp.eye(3) + 2 * mu * eps

        U_j = vmap(u_single)(jnp.array(X))
        S_j = vmap(sigma_single)(jnp.array(X))
        return np.array(U_j), np.array(S_j)

    else:
        # NumPy path: central differences for du/dx
        S = np.zeros((X.shape[0], 3, 3), dtype=float)
        for i in range(X.shape[0]):
            x0 = X[i]
            G = np.zeros((3, 3), dtype=float)
            for k in range(3):
                dx = np.zeros(3)
                dx[k] = fd_eps
                S0p, Qp = _integrate_S0_Q_np(x0 + dx)
                S0m, Qm = _integrate_S0_Q_np(x0 - dx)
                up = C1 * S0p * t + C2 * (Qp @ t)
                um = C1 * S0m * t + C2 * (Qm @ t)
                G[:, k] = (up - um) / (2 * fd_eps)
            eps = 0.5 * (G + G.T)
            tr_eps = np.trace(eps)
            S[i] = lam * tr_eps * np.eye(3) + 2 * mu * eps
        return U, S


# %% Main
# vertices (counterclockwise; orientation doesn't matter for this integral)
V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

t = np.array([0.0, 0.0, 1.0])  # constant traction (Pa)
mu = 30e9
nu = 0.25

# evaluation points (off the plane z=0)
X = np.array([[0.2, 0.2, 0.3], [2.0, -1.0, 0.5], [0.3, 0.1, 2.0]])

# NumPy backend (portable)
u, sigma = kelvin_triangle_constant_traction(
    X, V, t, mu, nu, n_subdiv=1, backend="numpy", return_sigma=False
)

# JAX backend (exact gradient for stress)
# (make sure JAX is installed; 64‑bit recommended)
# u_jax, sigma_jax = kelvin_triangle_constant_traction(X, V, t, mu, nu,
#                                                      n_subdiv=1, backend="jax",
#                                                      return_sigma=True)
