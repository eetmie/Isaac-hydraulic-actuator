"""USD-derived planar kinematics, conservative geometry and reach governing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .core import DEFAULT_ASSET, JOINT_NAMES, ControllerSettings, wrap_angle

# Encloses the home tip (598, 26 mm) and most of what is reachable at the home bucket angle.
DRAWING_BOX = np.array([0.32, 0.72, -0.30, 0.20], dtype=np.float64)
# Carriage-pitch extremes [rad] a drawable pose must also be solvable at, since the carriage rocks passively.
STROKE_PITCHES = (-0.052, 0.052)
TRACKED_POINTS = ("tip", "pivot")


class ExcavatorKinematics:
    """Read joint frames once; evaluate articulated kinematics in batched Torch."""

    def __init__(self, asset: str | Path = DEFAULT_ASSET, device: str = "cpu"):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from scipy.spatial import ConvexHull

        self.path = Path(asset).resolve()
        self.device = device
        stage = Usd.Stage.Open(str(self.path))
        if not stage or UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != "Z":
            raise ValueError("Expected a meter-scaled, Z-up excavator USD")
        self.frames = []
        self.links = []
        limits = {}
        chain = [
            "revolute_carriage",
            "revolute_carriage_roll",
            "revolute_carriage_pitch",
            "revolute_lift",
            "revolute_tilt",
            "revolute_tool",
            "fix_tool",
        ]

        def matrix(joint, suffix):
            r = joint.GetAttribute("physics:localRot" + suffix).Get()
            p = joint.GetAttribute("physics:localPos" + suffix).Get()
            row = Gf.Matrix4d().SetRotate(Gf.Quatd(r)) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(p))
            return torch.tensor(np.array(row).T.copy(), dtype=torch.float32, device=device)

        for name in chain:
            joint = stage.GetPrimAtPath("/excavator/Joints/" + name)
            channel = JOINT_NAMES.index(name) if name in JOINT_NAMES else -1
            if channel >= 0:
                if joint.GetAttribute("physics:axis").Get() != "Y":
                    raise ValueError("The planar controller expects Y-axis arm and carriage-pitch joints")
                limits[channel] = np.deg2rad(
                    [
                        joint.GetAttribute("physics:lowerLimit").Get(),
                        joint.GetAttribute("physics:upperLimit").Get(),
                    ]
                )
            self.frames.append((matrix(joint, "0"), torch.linalg.inv(matrix(joint, "1")), channel))
            self.links.append(str(UsdPhysics.Joint(joint).GetBody1Rel().GetTargets()[0]))
        self.limits = torch.tensor(
            np.array([limits[i] for i in range(4)]), dtype=torch.float32, device=device
        )
        tip = stage.GetPrimAtPath("/excavator/bucket/ee_tip")
        if not tip:
            raise ValueError("Asset is missing /excavator/bucket/ee_tip")
        self.tip = torch.tensor(
            np.array(UsdGeom.Xformable(tip).GetLocalTransformation()).T.copy(),
            dtype=torch.float32,
            device=device,
        )
        # Convex silhouettes of the actual colliders, not large axis-aligned link boxes.
        transforms, _ = self._transforms(torch.zeros(1, 4, device=device))
        self.polygons = []
        cache = UsdGeom.XformCache()
        body_names = ["/excavator/lower_carriage"] + self.links
        for idx, path in enumerate(body_names):
            prim = stage.GetPrimAtPath(path)
            points = []
            inverse = cache.GetLocalToWorldTransform(prim).GetInverse()
            for child in Usd.PrimRange(prim):
                if not child.IsA(UsdGeom.Mesh) or not child.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                mesh_points = np.array(UsdGeom.Mesh(child).GetPointsAttr().Get())
                transform = cache.GetLocalToWorldTransform(child) * inverse
                matrix_np = np.array(transform)
                points.extend(mesh_points @ matrix_np[:3, :3] + matrix_np[3, :3])
            if not points:
                continue
            local = torch.tensor(np.array(points), dtype=torch.float32, device=device)
            homogeneous = torch.cat((local, torch.ones(len(local), 1, device=device)), dim=1)
            projected = (homogeneous @ transforms[idx][0].T)[:, [0, 2]].cpu().numpy()
            hull = ConvexHull(projected).vertices
            self.polygons.append((path.rsplit("/", 1)[-1], idx, homogeneous[hull]))
        adjacent = {
            frozenset(pair)
            for pair in [
                ("lower_carriage", "upper_carriage"),
                ("upper_carriage", "lift_boom"),
                ("lift_boom", "tilt_boom"),
                ("tilt_boom", "tool_body"),
                ("tool_body", "bucket"),
                ("tilt_boom", "bucket"),
            ]
        }
        self.collision_pairs = [
            (i, j)
            for i in range(len(self.polygons))
            for j in range(i + 1, len(self.polygons))
            if frozenset((self.polygons[i][0], self.polygons[j][0])) not in adjacent
        ]

    def _transforms(self, q: torch.Tensor):
        n = len(q)
        transform = torch.eye(4, device=q.device).expand(n, 4, 4).clone()
        transforms = [transform]
        joint_frames = {}
        for parent, child_inverse, channel in self.frames:
            transform = transform @ parent
            if channel >= 0:
                joint_frames[channel] = transform
                angle = q[:, channel]
                c, s = angle.cos(), angle.sin()
                rotation = torch.eye(4, device=q.device).expand(n, 4, 4).clone()
                rotation[:, 0, 0] = c
                rotation[:, 2, 2] = c
                rotation[:, 0, 2] = s
                rotation[:, 2, 0] = -s
                transform = transform @ rotation
            transform = transform @ child_inverse
            transforms.append(transform)
        return transforms, joint_frames

    def to_dict(self) -> dict:
        """Export geometry in meters/radians for inference without USD or SciPy."""
        return {
            "version": 1,
            "frames": [[a.tolist(), b.tolist(), channel] for a, b, channel in self.frames],
            "links": self.links,
            "limits_rad": self.limits.tolist(),
            "tip": self.tip.tolist(),
            "polygons": [[name, index, corners.tolist()] for name, index, corners in self.polygons],
            "collision_pairs": self.collision_pairs,
        }

    @classmethod
    def from_dict(cls, geometry: dict, device: str = "cpu"):
        """Load an exported geometry contract [m, rad] without a USD installation."""
        if geometry["version"] != 1:
            raise ValueError("Unsupported exported geometry")
        self = cls.__new__(cls)
        self.device = device
        self.path = None

        def tensor(value):
            return torch.tensor(value, dtype=torch.float32, device=device)

        self.frames = [(tensor(a), tensor(b), int(channel)) for a, b, channel in geometry["frames"]]
        self.links = geometry["links"]
        self.limits = tensor(geometry["limits_rad"])
        self.tip = tensor(geometry["tip"])
        self.polygons = [(name, int(index), tensor(corners)) for name, index, corners in geometry["polygons"]]
        self.collision_pairs = [tuple(pair) for pair in geometry["collision_pairs"]]
        return self

    def pose_jacobian(self, q: torch.Tensor, point: str = "tip") -> tuple[torch.Tensor, torch.Tensor]:
        """Return tip pose [m, m, rad] and Jacobian with respect to four angles [rad].

        ``point="pivot"`` tracks the bucket joint instead of the tip: same bucket angle, and the bucket joint's
        column has no linear part, so linear and angular motion decouple (Egli & Hutter, RA-L 2022).
        """
        if point not in TRACKED_POINTS:
            raise ValueError(f"unknown tracked point {point!r}")
        transforms, joints = self._transforms(q)
        if point == "tip":
            point = (transforms[-1] @ self.tip)[:, :3, 3]
        else:
            point = joints[2][:, :3, 3]
        blade = transforms[-1][:, :3, 1]  # bucket-local +Y points toward the cutting lip
        angle = torch.atan2(-blade[:, 2], blade[:, 0])
        pose = torch.cat((point[:, ::2], angle[:, None]), dim=1)
        columns = []
        for i in range(4):
            frame = joints[i]
            axis = frame[:, :3, 1]
            linear = torch.linalg.cross(axis, point - frame[:, :3, 3])[:, ::2]
            columns.append(torch.cat((linear, axis[:, 1:2]), dim=1))
        return pose, torch.stack(columns, dim=-1)

    def tip_offset(self, angle: torch.Tensor) -> torch.Tensor:
        """Tip minus pivot [m] in X/Z [..., 2] for bucket angles [rad]; the bucket is one rigid body."""
        if not hasattr(self, "_tip_offset"):
            q = torch.zeros(1, 4, device=self.limits.device)
            tip, pivot = self.pose_jacobian(q)[0], self.pose_jacobian(q, "pivot")[0]
            self._tip_offset = (tip[0, :2] - pivot[0, :2], tip[0, 2])
        offset, angle0 = self._tip_offset
        turn = angle - angle0
        c, s = turn.cos(), turn.sin()  # rotation about +Y as in _transforms: x' = c x + s z, z' = -s x + c z
        return torch.stack((c * offset[0] + s * offset[1], -s * offset[0] + c * offset[1]), dim=-1)

    def build_collision_grid(
        self,
        resolution: int = 96,
        pitches: tuple[float, ...] = (-0.03, 0.0, 0.03),
        cache_dir: str | Path | None = None,
    ) -> None:
        """Tabulate self-collision over boom/arm/bucket so :meth:`colliding` becomes a lookup.

        A cell is marked colliding if the exact check fails at any listed carriage pitch [rad], then
        dilated by one cell so nearest-cell lookup stays conservative. The table is cached per asset.
        """
        from .core import ROOT, sha256

        cache_dir = Path(cache_dir) if cache_dir else ROOT / "runs/cache"
        key = f"collision_{sha256(self.path)[:16]}_{resolution}_{len(pitches)}.pt"
        path = cache_dir / key
        if path.is_file():
            grid = torch.load(path, map_location=self.device, weights_only=True)
        else:
            lo, hi = self.limits[:3, 0], self.limits[:3, 1]
            axes = [
                torch.linspace(float(lo[i]), float(hi[i]), resolution, device=self.device) for i in range(3)
            ]
            mesh = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
            grid = torch.zeros(len(mesh), dtype=torch.bool, device=self.device)
            for pitch in pitches:
                for begin in range(0, len(mesh), 16384):
                    chunk = mesh[begin : begin + 16384]
                    q = torch.cat((chunk, torch.full_like(chunk[:, :1], pitch)), dim=1)
                    grid[begin : begin + len(chunk)] |= self._colliding_exact(q)
            grid = grid.reshape(resolution, resolution, resolution)
            dilated = torch.nn.functional.max_pool3d(grid[None, None].float(), 3, stride=1, padding=1)
            grid = dilated[0, 0] > 0.5
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(grid.cpu(), path)
        self._grid = grid.to(self.device)

    def colliding(self, q: torch.Tensor) -> torch.Tensor:
        """Conservatively flag nonadjacent collider intersections in the arm plane."""
        grid = getattr(self, "_grid", None)
        if grid is None:
            return self._colliding_exact(q)
        resolution = grid.shape[0]
        lo, hi = self.limits[:3, 0], self.limits[:3, 1]
        index = ((q[:, :3] - lo) / (hi - lo) * (resolution - 1)).round().long().clamp(0, resolution - 1)
        return grid[index[:, 0], index[:, 1], index[:, 2]]

    def _colliding_exact(self, q: torch.Tensor) -> torch.Tensor:
        transforms, _ = self._transforms(q)
        polygons = [
            (corners @ transforms[idx].transpose(1, 2))[:, :, [0, 2]] for _, idx, corners in self.polygons
        ]
        collided = torch.zeros(len(q), dtype=torch.bool, device=q.device)
        for i, j in self.collision_pairs:
            a, b = polygons[i], polygons[j]
            edges = torch.cat((torch.roll(a, 1, 1) - a, torch.roll(b, 1, 1) - b), dim=1)
            axes = torch.stack((-edges[:, :, 1], edges[:, :, 0]), dim=-1)
            axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            pa, pb = a @ axes.transpose(1, 2), b @ axes.transpose(1, 2)
            separated = (pa.amax(1) <= pb.amin(1) + 0.001) | (pb.amax(1) <= pa.amin(1) + 0.001)
            collided |= ~separated.any(dim=1)
        return collided

    def valid(self, q: torch.Tensor, margin: float | None = None) -> torch.Tensor:
        """Check finite, in-limit, nonintersecting configurations [rad]."""
        # The passive pitch channel has its own measured +/-3 degree bound, not an arm margin.
        margin = getattr(self, "joint_margin", 0.06) if margin is None else margin
        margins = q.new_tensor([margin, margin, margin, 0.0])
        inside = ((q >= self.limits[:, 0] + margins) & (q <= self.limits[:, 1] - margins)).all(1)
        return torch.isfinite(q).all(1) & inside & ~self.colliding(q)

    def inverse(self, targets: torch.Tensor, seed: torch.Tensor, iterations: int = 35):
        """Solve fixed-carriage-pitch IK for poses [m, m, rad], starting at seed [rad]."""
        q = seed.expand(len(targets), 4).clone()
        weights = q.new_tensor([1.0, 1.0, 0.2])
        eye = torch.eye(3, device=q.device)
        for _ in range(iterations):
            pose, jac = self.pose_jacobian(q)
            error = targets - pose
            error[:, 2] = wrap_angle(error[:, 2])
            j = jac[:, :, :3] * weights[None, :, None]
            dq = torch.linalg.solve(
                j.transpose(1, 2) @ j + 1e-6 * eye, (j.transpose(1, 2) @ (error * weights)[:, :, None])
            ).squeeze(-1)
            q[:, :3] += dq.clamp(-0.2, 0.2)
            q[:, :3] = q[:, :3].clamp(self.limits[:3, 0], self.limits[:3, 1])
        pose, _ = self.pose_jacobian(q)
        reached = (pose[:, :2] - targets[:, :2]).norm(dim=1) < 0.0005
        reached &= wrap_angle(pose[:, 2] - targets[:, 2]).abs() < 0.005
        return q, reached & self.valid(q)

    def drawing_box(self, q: torch.Tensor) -> np.ndarray:
        """Return the operator-facing X/Z canvas; complete paths are validated separately."""
        del q
        return DRAWING_BOX.copy()

    def reachable(self, targets: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Flag poses [m, m, rad] solvable from ``q`` [rad] at its own and the extreme carriage pitches."""
        reachable = torch.ones(len(targets), dtype=torch.bool, device=targets.device)
        for pitch in (STROKE_PITCHES[0], float(q[0, 3]), STROKE_PITCHES[1]):
            seed = q[:1].clone()
            seed[:, 3] = pitch
            reachable &= self.inverse(targets, seed)[1]
        return reachable

    def reachable_cells(
        self, q: torch.Tensor, box: np.ndarray, angle: float, spacing: float = 0.005
    ) -> np.ndarray:
        """Tile the drawing box and flag the cells whose centers are :meth:`reachable` at a held bucket angle.

        Args:
            q: IK seed configuration [rad], shape [1, 4].
            box: Drawing rectangle ``[x_min, x_max, z_min, z_max]`` [m].
            angle: Held bucket angle [rad].
            spacing: Nominal cell size [m]; the cell count is rounded so cells tile the box exactly.

        Returns:
            Boolean array of shape [x cells, z cells], indexed from ``(x_min, z_min)``.
        """
        counts = (max(1, round((box[1] - box[0]) / spacing)), max(1, round((box[3] - box[2]) / spacing)))
        x = box[0] + (np.arange(counts[0]) + 0.5) * (box[1] - box[0]) / counts[0]
        z = box[2] + (np.arange(counts[1]) + 0.5) * (box[3] - box[2]) / counts[1]
        xx, zz = np.meshgrid(x, z, indexing="ij")
        targets = np.column_stack((xx.ravel(), zz.ravel(), np.full(xx.size, angle)))
        targets = torch.as_tensor(targets, dtype=q.dtype, device=q.device)
        return self.reachable(targets, q[:1]).reshape(counts).cpu().numpy()

    def local_reachable_box(self, q: torch.Tensor) -> np.ndarray:
        """Find a conservative pose-local rectangle for generated demonstration paths."""
        pose, _ = self.pose_jacobian(q[:1])
        offsets = torch.arange(-20, 21, device=q.device) * 0.005
        xx, zz = torch.meshgrid(offsets, offsets, indexing="ij")
        targets = pose.repeat(xx.numel(), 1)
        targets[:, 0] += xx.flatten()
        targets[:, 1] += zz.flatten()
        mask = torch.ones(len(targets), dtype=torch.bool, device=q.device)
        for pitch in (-0.052, 0.0, 0.052):
            seed = q[:1].clone()
            seed[:, 3] = pitch
            _, valid = self.inverse(targets, seed)
            mask &= valid
        mask = mask.reshape(41, 41).cpu().numpy()
        best = (0, 0)
        for hx in range(2, 21):
            for hz in range(2, 21):
                area = hx * hz
                if area > best[0] * best[1] and mask[20 - hx : 21 + hx, 20 - hz : 21 + hz].all():
                    best = (hx, hz)
        if min(best) < 2:
            raise ValueError("No reachable demonstration rectangle at the current pose")
        x, z = pose[0, :2].cpu().numpy()
        dx, dz = (np.array(best) - 1) * 0.005
        return np.array([x - dx, x + dx, z - dz, z + dz])


class CommandGovernor:
    """Project requested tip velocities into a conservative local reachable set.

    With ``joint_speed_limits`` [rad/s], shape [3, 2] = (negative, positive), the whole twist is also scaled down so
    no joint is asked for more than ``speed_margin`` of its full-valve speed; the direction of motion is preserved.
    """

    def __init__(
        self,
        kinematics: ExcavatorKinematics,
        settings: ControllerSettings,
        joint_speed_limits: torch.Tensor | None = None,
        speed_margin: float = 0.8,
        point: str = "tip",
    ):
        self.kin = kinematics
        self.cfg = settings
        self.joint_speed_limits = joint_speed_limits
        self.speed_margin = speed_margin
        self.point = point

    def __call__(self, q: torch.Tensor, v: torch.Tensor, requested: torch.Tensor):
        """Return admitted [m/s, m/s, rad/s] commands and intervention fractions."""
        cfg = self.cfg
        command = torch.nan_to_num(requested).clone()
        speed = command[:, :2].norm(dim=1, keepdim=True)
        command[:, :2] *= (cfg.speed_max / speed.clamp_min(1e-8)).clamp_max(1)
        command[:, 2].clamp_(-cfg.pitch_rate_max, cfg.pitch_rate_max)
        _, jac = self.kin.pose_jacobian(q) if self.point == "tip" else self.kin.pose_jacobian(q, self.point)
        weights = q.new_tensor([1.0, 1.0, 0.2])
        j = jac[:, :, :3] * weights[None, :, None]
        # Commands describe the arm-driven twist; passive carriage rocking is neither commanded nor tracked.
        rhs = command * weights
        dq = torch.linalg.solve(
            j.transpose(1, 2) @ j + 1e-5 * torch.eye(3, device=q.device), j.transpose(1, 2) @ rhs[:, :, None]
        ).squeeze(-1)
        if self.joint_speed_limits is not None:
            limits = self.joint_speed_limits.to(q.device)
            allowed = torch.where(dq < 0, limits[:, 0], limits[:, 1]) * self.speed_margin
            dq = dq * (allowed / dq.abs().clamp_min(1e-9)).amin(dim=1, keepdim=True).clamp_max(1)
        predicted_q = q[:, :3] + 0.25 * v[:, :3]
        low = ((self.kin.limits[:3, 0] + cfg.joint_margin - predicted_q) / cfg.lookahead).clamp_max(0)
        high = ((self.kin.limits[:3, 1] - cfg.joint_margin - predicted_q) / cfg.lookahead).clamp_min(0)
        dq = dq.clamp(low, high)
        # Backtrack proposed configurations against conservative collision geometry.
        scale = torch.ones(len(q), device=q.device)
        for _ in range(5):
            proposed = q.clone()
            proposed[:, :3] += cfg.lookahead * dq * scale[:, None]
            safe = self.kin.valid(proposed)
            scale = torch.where(safe, scale, scale * 0.5)
        proposed = q.clone()
        proposed[:, :3] += cfg.lookahead * dq * scale[:, None]
        scale = torch.where(self.kin.valid(proposed), scale, 0.0)
        admitted = torch.einsum("nij,nj->ni", jac[:, :, :3], dq * scale[:, None])
        # Never increase the requested translational magnitude during projection.
        admitted[:, :2] *= (speed / admitted[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
        admitted[:, 2].clamp_(-cfg.pitch_rate_max, cfg.pitch_rate_max)
        intervention = ((admitted - command) * weights).norm(dim=1) / (command * weights).norm(
            dim=1
        ).clamp_min(0.005)
        return admitted, intervention.clamp_max(1)
