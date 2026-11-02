"""Geometry helpers. All positions are mm. No implicit mesh repair or hole filling."""
import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError('Direction must be finite and nonzero')
    return v / length


def tool_frame(outward, tangent=(1., 0., 0.)):
    """Tool +Z approaches the surface; columns express tool axes in the current working frame."""
    z = -unit(outward)
    tangent = np.asarray(tangent, dtype=float)
    x = tangent - z * np.dot(tangent, z)
    if np.linalg.norm(x) < 1e-8:
        basis = np.eye(3)[np.argmin(np.abs(z))]
        x = basis - z * np.dot(basis, z)
    x = unit(x)
    return np.column_stack((x, np.cross(z, x), z))
