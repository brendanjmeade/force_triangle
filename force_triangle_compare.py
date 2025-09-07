# %% Imports
import math
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
import warnings
from scipy import integrate

# Suppress warnings about divide by zero and invalid values (expected near singularities)
warnings.filterwarnings("ignore", category=RuntimeWarning)

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")


# %% Functions
def _ln_sectan(u):
    # Stable: ln(sec u + tan u) = ln( tan(pi/4 + u/2) )
    return np.log(np.tan(0.25 * np.pi + 0.5 * u))


def _onplane_subtriangle_contrib(P, A, B, n_hat):
    """
    Closed-form contributions over the subtriangle (P, A, B),
    assuming P lies in the triangle plane. Returns (S0, Q_local, R)
    where Q_local is 2x2 in the local basis [e1,e2] and R is the 3x2
    matrix whose columns are [e1,e2] to rotate back: Q_global = R Q_local R^T.
    """
    a_vec = A - P
    b_vec = B - P
    ra = np.linalg.norm(a_vec)
    rb = np.linalg.norm(b_vec)
    if ra == 0.0 or rb == 0.0:
        # Degenerate; zero area
        R = np.column_stack((np.zeros(3), np.zeros(3)))
        return 0.0, np.zeros((2, 2)), R

    # Make local basis: e1 along a_vec, e2 = n_hat x e1 (right-handed in-plane)
    e1 = a_vec / ra
    e2 = np.cross(n_hat, e1)  # already unit: n_hat ⟂ e1
    # Ensure b is measured with positive oriented angle wrt [e1,e2]
    cb = np.dot(e1, b_vec) / rb  # cos Θ
    sb = np.dot(e2, b_vec) / rb  # sin Θ
    # If the angle is negative, swap A,B so that Θ>0
    if sb <= 0:
        a_vec, b_vec = b_vec, a_vec
        ra = np.linalg.norm(a_vec)
        rb = np.linalg.norm(b_vec)
        e1 = a_vec / ra
        e2 = np.cross(n_hat, e1)
        cb = np.dot(e1, b_vec) / rb
        sb = np.dot(e2, b_vec) / rb

    # Now Θ ∈ (0, π)
    Theta = np.arctan2(sb, cb)
    if Theta <= 1e-15:
        R = np.column_stack((e1, e2))
        return 0.0, np.zeros((2, 2)), R

    # Carley’s line x = r1 + a y parameter; φ = atan(a); β = r1*sqrt(1+a^2)
    a_param = (rb * cb - ra) / (rb * sb)
    phi = np.arctan(a_param)
    beta = ra * np.sqrt(1.0 + a_param * a_param)

    # Bounds in the shifted angle u = θ + φ
    uL = phi
    uU = Theta + phi

    # Primitive differences
    Ldiff = _ln_sectan(uU) - _ln_sectan(uL)  # ∫ sec u du
    Sdiff = np.sin(uU) - np.sin(uL)  # ∫ cos u du
    Cdiff = np.cos(uU) - np.cos(uL)  # ∫ -sin u du
    Secdiff = (1.0 / np.cos(uU)) - (1.0 / np.cos(uL))  # ∫ tan u sec u du

    # S0 = ∫ r(θ) dθ = β [ln(sec u + tan u)]_{uL}^{uU}
    S0 = beta * Ldiff

    # Q entries in local (e1,e2): Qxx = ∫ cos^2 θ r(θ) dθ, etc.
    c2phi = np.cos(2.0 * phi)
    s2phi = np.sin(2.0 * phi)
    sphi = np.sin(phi)
    cphi = np.cos(phi)

    # Ixx(u) primitive for β * ∫ sec u * cos^2 θ du
    # Ixx(u) = cos(2φ) sin u - sin(2φ) cos u + sin^2 φ * ln(sec u + tan u)
    # so ΔIxx = c2phi*Sdiff - s2phi*Cdiff + (sphi*sphi)*Ldiff
    Ixx_diff = c2phi * Sdiff - s2phi * Cdiff + (sphi * sphi) * Ldiff
    Qxx = beta * Ixx_diff
    # Qyy = β [ Ldiff - Ixx_diff ]
    Qyy = beta * (Ldiff - Ixx_diff)
    # Ixy(u) primitive for β * ∫ sec u * sin θ cos θ du
    # Ixy(u) = -cos(2φ) cos u + (sin φ cos φ) [ln(sec u + tan u) - sin u]
    Ixy_diff = (-c2phi) * Cdiff + (sphi * cphi) * (Ldiff - Sdiff)
    Qxy = beta * Ixy_diff

    Qloc = np.array([[Qxx, Qxy], [Qxy, Qyy]])
    R = np.column_stack((e1, e2))  # rotate 2D local to 3D
    return S0, Qloc, R


def kelvin_triangle_onfacet_u(P, V, traction, mu, nu, *, tol_planarity=1e-12):
    """
    Displacement u at a field point P lying on the (flat) triangular facet V (3x3),
    due to a constant traction vector 'traction' applied over the facet, in a full-space,
    homogeneous isotropic elastic medium (Kelvin single-layer).

    Closed form (no quadrature), valid for receivers in the triangle plane.
    Returns:
        u : (3,) ndarray
        (also returns S0 and Q if you want to inspect them)
    """
    V = np.asarray(V, dtype=float).reshape(3, 3)
    P = np.asarray(P, dtype=float).reshape(3)
    t = np.asarray(traction, dtype=float).reshape(3)

    # Triangle geometry and plane check
    e1 = V[1] - V[0]
    e2 = V[2] - V[0]
    n = np.cross(e1, e2)
    An2 = np.dot(n, n)
    if An2 == 0.0:
        raise ValueError("Degenerate triangle.")
    n_hat = n / np.sqrt(An2)
    # planarity check for P:
    d = np.dot(n_hat, P - V[0])
    if abs(d) > tol_planarity:
        raise ValueError(
            "Field point P is not on the triangle plane (|distance| > tol)."
        )

    # Cycle edges to tile the facet with three subtriangles (P, Va, Vb)
    edges = [(0, 1), (1, 2), (2, 0)]
    S0_total = 0.0
    Q_total = np.zeros((3, 3), dtype=float)

    for ia, ib in edges:
        S0, Qloc, R = _onplane_subtriangle_contrib(P, V[ia], V[ib], n_hat)
        # Accumulate scalar S0 and rotate 2x2 Qloc back to 3D:
        S0_total += S0
        Q_total += R @ Qloc @ R.T

    # Kelvin combination
    pre = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    u = pre * ((3.0 - 4.0 * nu) * S0_total * t + Q_total @ t)
    return u, S0_total, Q_total


# %% Numerical integration functions from force_triangle_numeric.py
def _dunavant_rule_deg5():
    """Dunavant degree-5 (7-point) quadrature on the reference triangle."""
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
    """Get triangle geometry."""
    e1 = V[1] - V[0]
    e2 = V[2] - V[0]
    n = np.cross(e1, e2)
    J = np.linalg.norm(n)  # |e1 x e2|  (twice area)
    if J == 0.0:
        raise ValueError("Degenerate triangle.")
    n_hat = n / J
    return e1, e2, n_hat, J


def _subdivide_triangle(V):
    """Subdivide triangle into 4 sub-triangles."""
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


def kelvin_triangle_numeric(X, V, traction, mu, nu, n_subdiv=1):
    """
    Numerical computation of displacement u(x) from a constant traction
    applied over a flat triangle using quadrature.

    Parameters:
        X : (N,3) array - Field points
        V : (3,3) array - Triangle vertices
        traction : (3,) array - Constant traction vector
        mu : float - Shear modulus
        nu : float - Poisson's ratio
        n_subdiv : int - Subdivision level (0-3 typical)

    Returns:
        U : (N,3) array - Displacements
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    V = np.asarray(V, dtype=float).reshape(3, 3)
    t = np.asarray(traction, dtype=float).reshape(3)

    # material constants
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

    # numeric integrator for S0 and Q
    def _integrate_S0_Q_np(x):
        S0 = 0.0
        Q = np.zeros((3, 3), dtype=float)
        for tri, e1, e2, J in tris_geom:
            Y = tri[0] + np.outer(xi, e1) + np.outer(eta, e2)  # (nq,3)
            R = x[None, :] - Y  # (nq,3)
            r = np.linalg.norm(R, axis=1)  # (nq,)
            # Handle singularities by adding small epsilon
            r = np.maximum(r, 1e-15)
            invr = 1.0 / r
            invr3 = invr**3
            fac = w * J
            S0 += np.sum(fac * invr)
            Q += np.einsum("q,qi,qj->ij", fac * invr3, R, R)
        return S0, Q

    # compute displacements
    U = np.zeros((X.shape[0], 3), dtype=float)
    for i in range(X.shape[0]):
        S0, Q = _integrate_S0_Q_np(X[i])
        U[i] = C1 * S0 * t + C2 * (Q @ t)

    return U


def kelvin_triangle_dblquad(X, V, traction, mu, nu):
    """
    Numerical computation of displacement u(x) from a constant traction
    applied over a flat triangle using scipy.integrate.dblquad.

    Parameters:
        X : (N,3) array - Field points
        V : (3,3) array - Triangle vertices
        traction : (3,) array - Constant traction vector
        mu : float - Shear modulus
        nu : float - Poisson's ratio

    Returns:
        U : (N,3) array - Displacements
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    V = np.asarray(V, dtype=float).reshape(3, 3)
    t = np.asarray(traction, dtype=float).reshape(3)

    # material constants
    C1 = (3 - 4 * nu) / (16 * math.pi * mu * (1 - nu))
    C2 = 1.0 / (16 * math.pi * mu * (1 - nu))

    # Triangle vertices
    v0, v1, v2 = V[0], V[1], V[2]

    # Compute displacements
    U = np.zeros((X.shape[0], 3), dtype=float)

    for idx in range(X.shape[0]):
        x_field = X[idx]

        # Define integrand functions for S0 and Q components
        def integrand_S0(xi, eta):
            # Convert parametric coords to physical coords
            # y = v0 + xi*(v1-v0) + eta*(v2-v0)
            y = v0 + xi * (v1 - v0) + eta * (v2 - v0)
            r_vec = x_field - y
            r = np.linalg.norm(r_vec)
            if r < 1e-15:
                return 0.0
            # Jacobian = ||(v1-v0) x (v2-v0)||
            jac = np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            return jac / r

        def integrand_Q(xi, eta, i, j):
            # Convert parametric coords to physical coords
            y = v0 + xi * (v1 - v0) + eta * (v2 - v0)
            r_vec = x_field - y
            r = np.linalg.norm(r_vec)
            if r < 1e-15:
                return 0.0
            jac = np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            return jac * r_vec[i] * r_vec[j] / (r**3)

        # Integration bounds for reference triangle (0,0), (1,0), (0,1)
        # eta from 0 to 1-xi, xi from 0 to 1
        def eta_lower(xi):
            return 0

        def eta_upper(xi):
            return 1 - xi

        # Compute S0 integral
        S0, _ = integrate.dblquad(
            integrand_S0, 0, 1, eta_lower, eta_upper, epsabs=1e-8, epsrel=1e-8
        )

        # Compute Q matrix components (only upper triangle due to symmetry)
        Q = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                Q[i, j], _ = integrate.dblquad(
                    lambda xi, eta: integrand_Q(xi, eta, i, j),
                    0,
                    1,
                    eta_lower,
                    eta_upper,
                    epsabs=1e-8,
                    epsrel=1e-8,
                )
                if i != j:
                    Q[j, i] = Q[i, j]  # Symmetry

        # Compute displacement
        U[idx] = C1 * S0 * t + C2 * (Q @ t)

    return U


# %% Main

# Triangle (counterclockwise) in z=0 plane
V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

# Observation coordinates
n = 31  # Reduced for faster dblquad computation
x = np.linspace(-0.5, 1.5, n)
y = np.linspace(-0.5, 1.5, n)
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()
z = np.zeros_like(x)
P = np.vstack((x, y, z)).T

# Constant traction (Pa)
t = np.array([1.0, 0.0, 0.0])

# Material propteries
mu = 30e9
nu = 0.25

# Storage for analytic output
ux_analytic = np.zeros_like(x)
uy_analytic = np.zeros_like(x)
uz_analytic = np.zeros_like(x)

# Calculate analytic displacements
print("Computing analytic solution...")
for i in range(len(P)):
    u, S0, Q = kelvin_triangle_onfacet_u(P[i, :], V, t, mu, nu)
    ux_analytic[i] = u[0]
    uy_analytic[i] = u[1]
    uz_analytic[i] = u[2]

# Calculate numerical displacements (dblquad)
print("Computing numerical solution (dblquad)...")
print(f"Computing for all {len(P)} points...")

# Initialize arrays for results
ux_dblquad = np.zeros_like(x)
uy_dblquad = np.zeros_like(x)
uz_dblquad = np.zeros_like(x)

# Compute for all points
for idx in range(len(P)):
    if idx % 100 == 0:  # Progress indicator every 100 points
        print(
            f"  Computing point {idx}/{len(P)} at ({P[idx, 0]:.2f}, {P[idx, 1]:.2f})..."
        )
    U_test = kelvin_triangle_dblquad(P[idx : idx + 1], V, t, mu, nu)
    ux_dblquad[idx] = U_test[0, 0]
    uy_dblquad[idx] = U_test[0, 1]
    uz_dblquad[idx] = U_test[0, 2]


# %% Plotting
xgrid = np.reshape(x, (n, n))
ygrid = np.reshape(y, (n, n))

# Reshape numerical solutions (dblquad)
ux_dblquad_grid = np.reshape(ux_dblquad, (n, n))
uy_dblquad_grid = np.reshape(uy_dblquad, (n, n))
uz_dblquad_grid = np.reshape(uz_dblquad, (n, n))

# Create comparison plots
fig = plt.figure(figsize=(20, 20))

plt.subplot(1, 3, 1)
plt.title("dblquad $u_x$")
plt.contourf(xgrid, ygrid, ux_dblquad_grid, levels=20)
plt.colorbar(label="$\\log_{10}|u_x|$")
plt.plot(
    [V[0, 0], V[1, 0], V[2, 0], V[0, 0]],
    [V[0, 1], V[1, 1], V[2, 1], V[0, 1]],
    "-k",
    linewidth=2,
)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.gca().set_aspect("equal")

plt.subplot(1, 3, 2)
plt.title("dblquad $u_y$")
plt.contourf(xgrid, ygrid, uy_dblquad_grid, levels=20)
plt.colorbar(label="$\\log_{10}|u_y|$")
plt.plot(
    [V[0, 0], V[1, 0], V[2, 0], V[0, 0]],
    [V[0, 1], V[1, 1], V[2, 1], V[0, 1]],
    "-k",
    linewidth=2,
)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.gca().set_aspect("equal")

plt.subplot(1, 3, 3)
plt.title("dblquad $u_z$")
plt.contourf(xgrid, ygrid, uz_dblquad_grid, levels=20)
plt.colorbar(label="$\\log_{10}|u_z|$")
plt.plot(
    [V[0, 0], V[1, 0], V[2, 0], V[0, 0]],
    [V[0, 1], V[1, 1], V[2, 1], V[0, 1]],
    "-k",
    linewidth=2,
)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.gca().set_aspect("equal")

# %%
