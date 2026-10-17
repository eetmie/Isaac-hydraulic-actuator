"""Quadratic MLS-MPM/APIC with Hencky Drucker-Prager soil and capsule-wall contact.

The soil's force on the prescribed bucket is the negative of its momentum change.
Torque is conjugate to CCW rotation in the x/z plot (negative world-Y rotation).
"""

import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.struct
class Parameters:
    dx: float
    inv_dx: float
    dt: float
    mass: float
    volume: float
    mu: float
    lam: float
    alpha: float
    cohesion: float  # Cohesive yield offset divided by 2 mu, in strain units.
    gravity: float
    floor_friction: float
    tool_friction: float
    xmax: float
    zmax: float
    radius: float
    tool_bound: float
    nx: int
    nz: int


@wp.func
def weights(f: float):
    return wp.vec3(0.5 * (1.5 - f) * (1.5 - f), 0.75 - (f - 1.0) * (f - 1.0), 0.5 * (f - 0.5) * (f - 0.5))


@wp.func
def project_strain(e: wp.vec2, mu: float, lam: float, alpha: float, cohesion: float):
    """Return to the DP cone, including its tensile apex; elastic strain is retained."""
    trace = e[0] + e[1]
    dev = e - wp.vec2(0.5 * trace, 0.5 * trace)
    norm = wp.length(dev)
    pressure_factor = (lam + mu) / mu
    apex = float(0.0)
    if alpha > 1e-8:
        apex = cohesion / (alpha * pressure_factor)
    if trace > apex:
        return wp.vec2(0.5 * apex, 0.5 * apex)
    excess = norm + alpha * pressure_factor * trace - cohesion
    if excess > 0.0 and norm > 1e-10:
        return e - (excess / norm) * dev
    return e


@wp.func
def local_point(x: wp.vec2, pose: wp.vec3):
    r = x - wp.vec2(pose[0], pose[1])
    c = wp.cos(pose[2])
    s = wp.sin(pose[2])
    return wp.vec2(c * r[0] + s * r[1], -s * r[0] + c * r[1])


@wp.func
def wall_distance(x: wp.vec2, pose: wp.vec3, walls: wp.array(dtype=wp.vec4), radius: float):
    q = local_point(x, pose)
    distance = float(1e10)
    normal = wp.vec2(0.0, 1.0)
    for k in range(walls.shape[0]):
        wall = walls[k]
        a = wp.vec2(wall[0], wall[1])
        b = wp.vec2(wall[2], wall[3])
        segment = b - a
        t = wp.clamp(wp.dot(q - a, segment) / wp.dot(segment, segment), 0.0, 1.0)
        delta = q - (a + t * segment)
        d = wp.length(delta)
        if d < distance:
            distance = d
            if d > 1e-8:
                normal = delta / d
            else:
                normal = wp.normalize(wp.vec2(-segment[1], segment[0]))
    c = wp.cos(pose[2])
    s = wp.sin(pose[2])
    return distance - radius, wp.vec2(c * normal[0] - s * normal[1], s * normal[0] + c * normal[1])


@wp.func
def tool_velocity(x: wp.vec2, pose: wp.vec3, twist: wp.vec3):
    r = x - wp.vec2(pose[0], pose[1])
    return wp.vec2(twist[0] - twist[2] * r[1], twist[1] + twist[2] * r[0])


@wp.func
def contact_velocity(v: wp.vec2, surface_v: wp.vec2, normal: wp.vec2, friction: float):
    relative = v - surface_v
    vn = wp.dot(relative, normal)
    result = v
    if vn < 0.0:
        tangent = relative - vn * normal
        speed = wp.length(tangent)
        change = -vn * normal
        if speed > 1e-10:
            change -= wp.min(friction * (-vn), speed) * tangent / speed
        result += change
    return result


@wp.func
def wrench(impulse: wp.vec2, point: wp.vec2, origin: wp.vec2):
    r = point - origin
    return wp.vec3(impulse[0], impulse[1], r[0] * impulse[1] - r[1] * impulse[0])


@wp.kernel
def clear_grid(
    mass: wp.array3d(dtype=float),
    momentum: wp.array3d(dtype=wp.vec2),
    velocity: wp.array3d(dtype=wp.vec2),
    pose: wp.array(dtype=wp.vec3),
    twist: wp.array(dtype=wp.vec3),
    cfg: Parameters,
):
    env, i, j = wp.tid()
    mass[env, i, j] = 0.0
    momentum[env, i, j] = wp.vec2(0.0)
    velocity[env, i, j] = wp.vec2(0.0)
    if i == 0 and j == 0:
        pose[env] = pose[env] + cfg.dt * twist[env]


@wp.kernel
def p2g(
    x: wp.array2d(dtype=wp.vec2),
    v: wp.array2d(dtype=wp.vec2),
    deformation: wp.array2d(dtype=wp.mat22),
    affine_v: wp.array2d(dtype=wp.mat22),
    mass: wp.array3d(dtype=float),
    momentum: wp.array3d(dtype=wp.vec2),
    faults: wp.array(dtype=int),
    cfg: Parameters,
):
    env, p = wp.tid()
    identity = wp.mat22(1.0, 0.0, 0.0, 1.0)
    f = (identity + cfg.dt * affine_v[env, p]) * deformation[env, p]
    determinant = wp.determinant(f)
    if not wp.isfinite(determinant) or determinant <= 1e-8:
        wp.atomic_add(faults, env, 1)
        f = identity  # Keep buffers readable; the host rejects the failed run.
    u, sigma, rotation_v = wp.svd2(f)
    strain = wp.vec2(wp.log(wp.max(wp.abs(sigma[0]), 1e-6)), wp.log(wp.max(wp.abs(sigma[1]), 1e-6)))
    elastic = project_strain(strain, cfg.mu, cfg.lam, cfg.alpha, cfg.cohesion)
    deformation[env, p] = (
        u * wp.diag(wp.vec2(wp.exp(elastic[0]), wp.exp(elastic[1]))) * wp.transpose(rotation_v)
    )
    principal_stress = 2.0 * cfg.mu * elastic + cfg.lam * (elastic[0] + elastic[1]) * wp.vec2(1.0)
    tau = u * wp.diag(principal_stress) * wp.transpose(u)
    affine = cfg.mass * affine_v[env, p] - (cfg.dt * cfg.volume * 4.0 * cfg.inv_dx * cfg.inv_dx) * tau
    grid_x = x[env, p] * cfg.inv_dx
    bi = int(wp.floor(grid_x[0] - 0.5))
    bj = int(wp.floor(grid_x[1] - 0.5))
    fx = grid_x - wp.vec2(float(bi), float(bj))
    wx = weights(fx[0])
    wz = weights(fx[1])
    for i in range(3):
        for j in range(3):
            ni = bi + i
            nj = bj + j
            if ni >= 0 and ni <= cfg.nx and nj >= 0 and nj <= cfg.nz:
                weight = wx[i] * wz[j]
                delta = (wp.vec2(float(i), float(j)) - fx) * cfg.dx
                wp.atomic_add(mass, env, ni, nj, weight * cfg.mass)
                wp.atomic_add(momentum, env, ni, nj, weight * (cfg.mass * v[env, p] + affine * delta))
            else:
                wp.atomic_add(faults, env, 1)


@wp.kernel
def grid_update(
    mass: wp.array3d(dtype=float),
    momentum: wp.array3d(dtype=wp.vec2),
    velocity: wp.array3d(dtype=wp.vec2),
    pose: wp.array(dtype=wp.vec3),
    twist: wp.array(dtype=wp.vec3),
    enabled: wp.array(dtype=int),
    walls: wp.array(dtype=wp.vec4),
    tool_impulse: wp.array(dtype=wp.vec3),
    boundary_impulse: wp.array(dtype=wp.vec3),
    cfg: Parameters,
):
    env, i, j = wp.tid()
    m = mass[env, i, j]
    if m > 1e-12:
        point = wp.vec2(float(i) * cfg.dx, float(j) * cfg.dx)
        value = momentum[env, i, j] / m + wp.vec2(0.0, -cfg.dt * cfg.gravity)
        origin = wp.vec2(pose[env][0], pose[env][1])
        if enabled[env] != 0 and wp.length(point - origin) <= cfg.tool_bound:
            distance, normal = wall_distance(point, pose[env], walls, cfg.radius)
            if distance <= 0.0:
                previous = value
                value = contact_velocity(
                    value, tool_velocity(point, pose[env], twist[env]), normal, cfg.tool_friction
                )
                wp.atomic_add(tool_impulse, env, wrench(-m * (value - previous), point, origin))
        previous = value
        if j <= 2 and value[1] < 0.0:
            value = contact_velocity(value, wp.vec2(0.0), wp.vec2(0.0, 1.0), cfg.floor_friction)
        if i <= 2 and value[0] < 0.0:
            value = wp.vec2(0.0, value[1])
        if i >= cfg.nx - 2 and value[0] > 0.0:
            value = wp.vec2(0.0, value[1])
        if j >= cfg.nz - 2 and value[1] > 0.0:
            value = wp.vec2(value[0], 0.0)
        wp.atomic_add(boundary_impulse, env, wrench(-m * (value - previous), point, wp.vec2(0.0)))
        velocity[env, i, j] = value


@wp.kernel
def g2p(
    x: wp.array2d(dtype=wp.vec2),
    v: wp.array2d(dtype=wp.vec2),
    affine_v: wp.array2d(dtype=wp.mat22),
    velocity: wp.array3d(dtype=wp.vec2),
    pose: wp.array(dtype=wp.vec3),
    twist: wp.array(dtype=wp.vec3),
    enabled: wp.array(dtype=int),
    walls: wp.array(dtype=wp.vec4),
    tool_impulse: wp.array(dtype=wp.vec3),
    boundary_impulse: wp.array(dtype=wp.vec3),
    penetration: wp.array(dtype=float),
    faults: wp.array(dtype=int),
    cfg: Parameters,
):
    env, p = wp.tid()
    old_x = x[env, p]
    grid_x = old_x * cfg.inv_dx
    bi = int(wp.floor(grid_x[0] - 0.5))
    bj = int(wp.floor(grid_x[1] - 0.5))
    fx = grid_x - wp.vec2(float(bi), float(bj))
    wx = weights(fx[0])
    wz = weights(fx[1])
    value = wp.vec2(0.0)
    c = wp.mat22(0.0)
    for i in range(3):
        for j in range(3):
            ni = bi + i
            nj = bj + j
            if ni >= 0 and ni <= cfg.nx and nj >= 0 and nj <= cfg.nz:
                weight = wx[i] * wz[j]
                node_v = velocity[env, ni, nj]
                delta = (wp.vec2(float(i), float(j)) - fx) * cfg.dx
                value += weight * node_v
                c += (4.0 * cfg.inv_dx * cfg.inv_dx * weight) * wp.outer(node_v, delta)
    point = old_x + cfg.dt * value
    origin = wp.vec2(pose[env][0], pose[env][1])
    if enabled[env] != 0 and wp.length(point - origin) <= cfg.tool_bound:
        distance, normal = wall_distance(point, pose[env], walls, cfg.radius)
        if distance < 0.0:
            wp.atomic_max(penetration, env, -distance)
            point -= distance * normal
            previous = value
            value = contact_velocity(
                value, tool_velocity(point, pose[env], twist[env]), normal, cfg.tool_friction
            )
            wp.atomic_add(tool_impulse, env, wrench(-cfg.mass * (value - previous), point, origin))
            c = wp.mat22(0.0)  # A particle contact invalidates the interpolated local affine field.
    previous = value
    minimum = 2.0 * cfg.dx
    clipped = wp.vec2(
        wp.clamp(point[0], minimum, cfg.xmax - minimum), wp.clamp(point[1], minimum, cfg.zmax - minimum)
    )
    if wp.length(clipped - point) > cfg.dx:
        wp.atomic_add(faults, env, 1)
    if clipped[0] != point[0]:
        value = wp.vec2(0.0, value[1])
        c = wp.mat22(0.0)
    if clipped[1] != point[1]:
        value = wp.vec2(value[0], 0.0)
        c = wp.mat22(0.0)
    wp.atomic_add(boundary_impulse, env, wrench(-cfg.mass * (value - previous), clipped, wp.vec2(0.0)))
    if (
        not wp.isfinite(value[0])
        or not wp.isfinite(value[1])
        or not wp.isfinite(clipped[0])
        or not wp.isfinite(clipped[1])
    ):
        wp.atomic_add(faults, env, 1)
    x[env, p] = clipped
    v[env, p] = value
    affine_v[env, p] = c
