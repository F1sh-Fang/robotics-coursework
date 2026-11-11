"""Current project's spherical-wrist model; distances mm, internal angles rad.

The original TCP is the grip point. The pen tip is +tip_offset along tool Z;
R_HOLD maps that direction to world -Z. No robot/controller calibration is implied.
"""
from dataclasses import dataclass
import numpy as np

LENGTHS = np.array([491., 450., 450., 84.])
R_HOLD = np.diag([-1., 1., -1.])
LIMITS_DEG = np.array([[-170, 170], [-120, 120], [-140, 140],
                       [-170, 170], [-120, 120], [-360, 360]], dtype=float)


def tip_to_grip(tip, tip_offset):
    return np.asarray(tip, dtype=float) - R_HOLD @ np.array([0., 0., tip_offset])


def forward_kinematics(q):
    """POE forward kinematics for the project's structured spherical-wrist robot."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    if q.shape[1] != 6 or not np.isfinite(q).all():
        raise ValueError("Expected finite N x 6 joint angles in radians")
    n = len(q)
    rotation = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    position = np.zeros((n, 3))
    l1, l2, l3, l4 = LENGTHS
    heights = [0, l1, l1+l2, l1+l2+l3, l1+l2+l3, l1+l2+l3]
    for j, axis in enumerate(('z', 'y', 'y', 'z', 'y', 'z')):
        w = np.array([0., 0., 1.]) if axis == 'z' else np.array([0., 1., 0.])
        k = np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])
        angle = q[:, j, None, None]
        r = np.eye(3) + np.sin(angle)*k + (1-np.cos(angle))*(k@k)
        p = np.einsum('nij,j->ni', np.eye(3)-r, [0., 0., heights[j]])
        position += np.einsum('nij,nj->ni', rotation, p)
        rotation = rotation @ r
    position += np.einsum('nij,j->ni', rotation, [0., 0., l1+l2+l3+l4])
    return position, rotation @ np.diag([-1., -1., 1.])


def forward_tip(q, tip_offset):
    grip, rotations = forward_kinematics(q)
    tip = grip + rotations[:, :, 2]*tip_offset
    return tip, grip, rotations


def branch_angles(tip, tip_offset, shoulder, elbow, wrist=1):
    """Closed form for R_HOLD; preserves branch identity over the entire path."""
    grip = np.atleast_2d(tip_to_grip(tip, tip_offset))
    l1, l2, l3, l4 = LENGTHS
    r = np.hypot(grip[:, 0], grip[:, 1])
    h = grip[:, 2] + l4 - l1
    if np.any(r < 1e-6):
        raise ValueError('Path crosses the base-axis singularity')
    c3 = (r*r+h*h-l2*l2-l3*l3)/(2*l2*l3)
    if np.any(np.abs(c3) > 1+1e-10):
        raise ValueError('Pen path is outside the arm reach')
    q1 = np.arctan2(grip[:, 1], grip[:, 0]) + (np.pi if shoulder < 0 else 0)
    q3 = elbow*np.arccos(np.clip(c3, -1, 1))
    q2 = np.arctan2(shoulder*r, h)-np.arctan2(l3*np.sin(q3), l2+l3*np.cos(q3))
    q4 = np.zeros(len(r))
    q5 = np.pi-q2-q3
    q6 = q1-np.pi
    if wrist < 0:
        q4 += np.pi
        q5 *= -1
        q6 += np.pi
    q = np.column_stack([q1, q2, q3, q4, q5, q6])
    # Unwrapping is allowed only within a fixed analytic branch, not across flips.
    return np.unwrap(np.arctan2(np.sin(q), np.cos(q)), axis=0)


@dataclass
class IKBranch:
    shoulder: int
    elbow: int
    wrist: int
    q_start: np.ndarray
    limits: np.ndarray
    tip_offset: float

    def solve(self, tip, previous=None):
        q = branch_angles(tip, self.tip_offset, self.shoulder, self.elbow, self.wrist)
        reference = self.q_start if previous is None else previous
        q += 2*np.pi*np.round((reference-q[0])/(2*np.pi))
        if np.any(q < self.limits[:, 0]-1e-10) or np.any(q > self.limits[:, 1]+1e-10):
            raise ValueError('Selected IK branch exceeds joint limits')
        if np.min(np.abs(np.sin(q[:, 2]))) < 1e-3 or np.min(np.abs(np.sin(q[:, 4]))) < 1e-3:
            raise ValueError('Path approaches an elbow/wrist singularity; change placement')
        if len(q) > 1 and np.max(np.abs(np.diff(q, axis=0))) > np.pi/2:
            raise ValueError('Discontinuous IK branch; refusing joint interpolation')
        return q


def select_branch(tip, tip_offset, limits_deg=LIMITS_DEG, seed_deg=None):
    """Preflight every branch over ordered path samples, then lock one branch."""
    limits = np.deg2rad(np.asarray(limits_deg, dtype=float))
    seed = np.zeros(6) if seed_deg is None else np.deg2rad(seed_deg)
    if seed.shape != (6,) or not np.isfinite(seed).all():
        raise ValueError('IK seed must contain six finite joint angles')
    candidates = []
    for shoulder in (1, -1):
        for elbow in (1, -1):
            for wrist in (1, -1):
                try:
                    q = branch_angles(tip, tip_offset, shoulder, elbow, wrist)
                    for j in range(6):
                        low = int(np.ceil((limits[j, 0]-q[:, j].min())/(2*np.pi)-1e-12))
                        high = int(np.floor((limits[j, 1]-q[:, j].max())/(2*np.pi)+1e-12))
                        if low > high:
                            raise ValueError('No legal full-path angle representation')
                        turn = int(np.clip(np.round((seed[j]-q[0, j])/(2*np.pi)), low, high))
                        q[:, j] += 2*np.pi*turn
                    branch = IKBranch(shoulder, elbow, wrist, q[0], limits, tip_offset)
                    q = branch.solve(tip)
                    margin = min(np.min(q-limits[:, 0]), np.min(limits[:, 1]-q))
                    cost = np.sum((q[0]-seed)**2) + 0.05/(margin+0.01)
                    candidates.append((cost, branch))
                except ValueError:
                    continue
    if not candidates:
        raise ValueError('No single nonsingular IK branch covers the whole pen path within limits')
    return min(candidates, key=lambda item: item[0])[1]
