"""Generate a small reproducible (t,z,y,x) wind-field example."""
from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)
t, z, y, x = 6, 8, 24, 24
dx = dy = 30.0
dt = 10.0
xx, yy = np.meshgrid(np.arange(x) * dx, np.arange(y) * dy)
dem = 20.0 + 8.0 * np.sin(xx / 250.0) * np.cos(yy / 300.0)
z_abs = np.arange(z) * 20.0 + 60.0
u = np.empty((t, z, y, x))
v = np.empty_like(u)
w = np.empty_like(u)
for it in range(t):
    for iz, height in enumerate(z_abs):
        phase = 0.08 * it
        u[it, iz] = 4.0 * (height / 100.0) ** 0.18 + 0.25 * np.sin(xx / 180 + phase)
        v[it, iz] = 1.2 + 0.18 * np.cos(yy / 210 + phase)
        dHdy, dHdx = np.gradient(dem, dy, dx)
        w[it, iz] = u[it, iz] * dHdx + v[it, iz] * dHdy
u += rng.normal(0, 0.025, u.shape)
v += rng.normal(0, 0.025, v.shape)
w += rng.normal(0, 0.012, w.shape)
air_mask = z_abs[:, None, None] > dem[None, :, :]
out = Path(__file__).with_name("example_wind.npz")
np.savez_compressed(out, u=u, v=v, w=w, dem_m=dem, air_mask=air_mask,
                    z_abs_m=z_abs, dx_m=dx, dy_m=dy, dt_s=dt)
print(out)

