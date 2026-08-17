import numpy as np
import matplotlib.pyplot as plt

# def _normalize_points(pts):
#     pts = np.asarray(pts)
#     if pts.ndim == 1 and pts.size == 3:
#         pts = pts.reshape(1, 3)
#     if pts.ndim == 3:  # (n_traj, T, 3) -> flatten
#         pts = pts.reshape(-1, 3)
#     norms = np.linalg.norm(pts, axis=1, keepdims=True)
#     norms[norms == 0] = 1.0
#     return pts / norms

def plot_on_sphere(demonstrations=None, initials=None, trajectory=None, goal=None, savepath=None, title=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # sphere mesh
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    uu, vv = np.meshgrid(u, v)
    xs = np.cos(uu) * np.sin(vv)
    ys = np.sin(uu) * np.sin(vv)
    zs = np.cos(vv)
    ax.plot_surface(xs, ys, zs, color='lightgrey', alpha=0.25, linewidth=0)

    if demonstrations is not None:
        for z in range(demonstrations.shape[0]):
            demo_pts = demonstrations[z, :, :]
            ax.scatter(demo_pts[:, 0], demo_pts[:, 1], demo_pts[:, 2], c='C0', s=6, label='demonstrations', alpha=0.8)

    if initials is not None:
        init_pts = initials
        ax.scatter(init_pts[:, 0], init_pts[:, 1], init_pts[:, 2], c='C2', s=30, label='initials', marker='x')

    if trajectory is not None:
        for z in range(trajectory.shape[0]):
            traj = np.asarray(trajectory[z, :, :])
            # extract first 3 entries if state has more (e.g. position + quat)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='C3', lw=1.5, label='simulated trajectory')
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='C3', s=40, marker='o', label='traj start')
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='C4', s=40, marker='*', label='traj end')

    if goal is not None:
        g = np.asarray(goal)
        if g.size > 3:
            g = g[:3]
        g = g / (np.linalg.norm(g) + 1e-12)
        ax.scatter([g[0]], [g[1]], [g[2]], c='r', s=80, marker='*', label='goal')

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    # ax.legend(loc='upper right')
    if savepath:
        fig.savefig(savepath, dpi=200)
    plt.show()