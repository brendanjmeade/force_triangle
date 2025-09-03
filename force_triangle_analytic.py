# %% Imports
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np

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


# %% Main

# Triangle (counterclockwise) in z=0 plane
V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

# Observation coordinates
n = 100
x = np.linspace(-0.5, 1.5, 100)
y = np.linspace(-0.5, 1.5, 100)
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

# Storage for output
ux = np.zeros_like(x)
uy = np.zeros_like(x)
uz = np.zeros_like(x)

# Calculate displacments.
for i in range(len(P)):
    u, S0, Q = kelvin_triangle_onfacet_u(P[i, :], V, t, mu, nu)
    ux[i] = u[0]
    uy[i] = u[1]
    uz[i] = u[2]

# Set nans to zero
ux = np.nan_to_num(ux, nan=0.0)
uy = np.nan_to_num(uy, nan=0.0)
uz = np.nan_to_num(uz, nan=0.0)


# %% Plotting
xgrid = np.reshape(x, (n, n))
ygrid = np.reshape(y, (n, n))
uxgrid = np.reshape(ux, (n, n))
uygrid = np.reshape(uy, (n, n))
uzgrid = np.reshape(uz, (n, n))

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.title("$u_x$")
plt.contourf(xgrid, ygrid, np.log10(np.abs(uxgrid)))
plt.plot(
    [V[0, 0], V[1, 0], V[2, 0], V[0, 0]], [V[0, 1], V[1, 1], V[2, 1], V[0, 1]], "-k"
)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.gca().set_aspect("equal")

plt.subplot(1, 3, 2)
plt.title("$u_y$")
plt.contourf(xgrid, ygrid, np.log10(np.abs(uygrid)))
plt.plot(
    [V[0, 0], V[1, 0], V[2, 0], V[0, 0]], [V[0, 1], V[1, 1], V[2, 1], V[0, 1]], "-k"
)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.gca().set_aspect("equal")

plt.subplot(1, 3, 3)
plt.title("$u_z$")
plt.contourf(xgrid, ygrid, np.log10(np.abs(uzgrid)))
plt.plot(
    [V[0, 0], V[1, 0], V[2, 0], V[0, 0]], [V[0, 1], V[1, 1], V[2, 1], V[0, 1]], "-k"
)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.gca().set_aspect("equal")

plt.show()

# %%
