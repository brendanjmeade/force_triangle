# %% Imports
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
import warnings
from scipy import integrate

# Suppress warnings about divide by zero and invalid values (expected near singularities)
warnings.filterwarnings("ignore", category=RuntimeWarning)

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")


# %% Functions

# def kelvin_triangle_onfacet_u(P, V, traction, mu, nu, *, tol_planarity=1e-12):
#     """
#     Displacement u at a field point P lying on the (flat) triangular facet V (3x3),
#     due to a constant traction vector 'traction' applied over the facet, in a full-space,
#     homogeneous isotropic elastic medium (Kelvin single-layer).

#     Closed form (no quadrature), valid for receivers in the triangle plane.
#     Returns:
#         u : (3,) ndarray
#         (also returns S0 and Q if you want to inspect them)
#     """
#     V = np.asarray(V, dtype=float).reshape(3, 3)
#     P = np.asarray(P, dtype=float).reshape(3)
#     t = np.asarray(traction, dtype=float).reshape(3)

#     # Triangle geometry and plane check
#     e1 = V[1] - V[0]
#     e2 = V[2] - V[0]
#     n = np.cross(e1, e2)
#     An2 = np.dot(n, n)
#     if An2 == 0.0:
#         raise ValueError("Degenerate triangle.")
#     n_hat = n / np.sqrt(An2)
#     # planarity check for P:
#     d = np.dot(n_hat, P - V[0])
#     if abs(d) > tol_planarity:
#         raise ValueError(
#             "Field point P is not on the triangle plane (|distance| > tol)."
#         )

#     # Cycle edges to tile the facet with three subtriangles (P, Va, Vb)
#     edges = [(0, 1), (1, 2), (2, 0)]
#     S0_total = 0.0
#     Q_total = np.zeros((3, 3), dtype=float)

#     for ia, ib in edges:
#         S0, Qloc, R = _onplane_subtriangle_contrib(P, V[ia], V[ib], n_hat)
#         # Accumulate scalar S0 and rotate 2x2 Qloc back to 3D:
#         S0_total += S0
#         Q_total += R @ Qloc @ R.T

#     # Kelvin combination
#     pre = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
#     u = pre * ((3.0 - 4.0 * nu) * S0_total * t + Q_total @ t)
#     return u, S0_total, Q_total


# %% Numerical integration functions from force_triangle_numeric.py
# def _dunavant_rule_deg5():
#     """Dunavant degree-5 (7-point) quadrature on the reference triangle."""
#     xi, eta, w = [], [], []
#     # centroid
#     xi.append(1 / 3)
#     eta.append(1 / 3)
#     w.append(0.225000000000000)
#     # group 2
#     a, b, w2 = 0.059715871789770, 0.470142064105115, 0.132394152788506
#     for alpha, beta, gamma in [(a, b, b), (b, b, a), (b, a, b)]:
#         xi.append(beta)
#         eta.append(gamma)
#         w.append(w2)
#     # group 3
#     a, b, w3 = 0.797426985353087, 0.101286507323456, 0.125939180544827
#     for alpha, beta, gamma in [(a, b, b), (b, b, a), (b, a, b)]:
#         xi.append(beta)
#         eta.append(gamma)
#         w.append(w3)
#     w = 0.5 * np.array(w)  # scale to half-area of reference triangle
#     return np.array(xi), np.array(eta), w


# def _triangle_geometry(V):
#     """Get triangle geometry."""
#     e1 = V[1] - V[0]
#     e2 = V[2] - V[0]
#     n = np.cross(e1, e2)
#     J = np.linalg.norm(n)  # |e1 x e2|  (twice area)
#     if J == 0.0:
#         raise ValueError("Degenerate triangle.")
#     n_hat = n / J
#     return e1, e2, n_hat, J


# def _subdivide_triangle(V):
#     """Subdivide triangle into 4 sub-triangles."""
#     m01 = 0.5 * (V[0] + V[1])
#     m12 = 0.5 * (V[1] + V[2])
#     m20 = 0.5 * (V[2] + V[0])
#     return np.array(
#         [
#             [V[0], m01, m20],
#             [m01, V[1], m12],
#             [m20, m12, V[2]],
#             [m01, m12, m20],
#         ]
#     )


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
    C1 = (3 - 4 * nu) / (16 * np.pi * mu * (1 - nu))
    C2 = 1.0 / (16 * np.pi * mu * (1 - nu))

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

# Initialize arrays for results
ux_dblquad = np.zeros_like(x)
uy_dblquad = np.zeros_like(x)
uz_dblquad = np.zeros_like(x)

# Compute for all points
for idx in range(len(P)):
    if idx % 10 == 0:  # Progress indicator every 10 points
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
plt.colorbar(label="$u_x$", fraction=0.046, pad=0.04)
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
plt.colorbar(label="$u_y$", fraction=0.046, pad=0.04)
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
plt.colorbar(label="$u_z$", fraction=0.046, pad=0.04)
plt.plot(
    [V[0, 0], V[1, 0], V[2, 0], V[0, 0]],
    [V[0, 1], V[1, 1], V[2, 1], V[0, 1]],
    "-k",
    linewidth=2,
)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.gca().set_aspect("equal")
