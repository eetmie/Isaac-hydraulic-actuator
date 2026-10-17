"""Small offline previews and an optional native Matplotlib viewer."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import Scene


def bucket_vertices(scene: Scene, pose) -> np.ndarray:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.array([[c, -s], [s, c]])
    return np.asarray(scene.bucket.vertices) @ rotation.T + np.asarray(pose[:2])


def image(
    scene: Scene, positions, initial, pose, *, time_s: float, phase: str, force, bucket_enabled: bool = True
) -> Image.Image:
    width, height = 960, 650
    canvas = Image.new("RGB", (width, height), "#f8f6f0")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 23)
    except OSError:
        font = title_font = ImageFont.load_default()
    scale = min((width - 90) / scene.width, (height - 140) / scene.height)
    x0, z0 = 45, height - 65

    def pixel(point):
        return (x0 + float(point[0]) * scale, z0 - float(point[1]) * scale)

    draw.text((35, 16), "2D soil  /  prescribed bucket", fill="#153943", font=title_font)
    draw.text(
        (35, 48),
        f"{time_s:5.2f} s   {phase}    Fx {force[0]:+.1f} N    Fz {force[1]:+.1f} N    torque {force[2]:+.2f} N m",
        fill="#35525b",
        font=font,
    )
    for x in np.arange(0, scene.width + 1e-6, 0.1):
        a, b = pixel((x, 0)), pixel((x, scene.height))
        draw.line([a, b], fill="#e4e3db")
    for z in np.arange(0, scene.height + 1e-6, 0.1):
        a, b = pixel((0, z)), pixel((scene.width, z))
        draw.line([a, b], fill="#e4e3db")
    floor = 2 * scene.dx
    draw.line([pixel((floor, floor)), pixel((scene.width - floor, floor))], fill="#8b8e82", width=3)
    colors = np.array([(123, 83, 45), (159, 109, 57), (192, 145, 83), (216, 175, 111)], dtype=np.uint8)
    layers = np.minimum(
        3, (4 * (initial[:, 1] - scene.soil_min[1]) / (scene.soil_max[1] - scene.soil_min[1])).astype(int)
    )
    radius = max(1.2, 0.30 * scene.dx / scene.particles_per_axis * scale)
    for point, layer in zip(positions, layers):
        px, pz = pixel(point)
        draw.ellipse((px - radius, pz - radius, px + radius, pz + radius), fill=tuple(colors[layer]))
    vertices = [pixel(p) for p in bucket_vertices(scene, pose)] if bucket_enabled else []
    for physical_radius, color in (
        (scene.collision_radius, "#b4d1d5"),
        (scene.bucket.wall_thickness / 2, "#1b5264"),
    ):
        if not vertices:
            break
        thickness = max(1, round(2 * physical_radius * scale))
        draw.line(vertices, fill=color, width=thickness, joint="curve")
        for px, pz in vertices:
            r = thickness / 2
            draw.ellipse((px - r, pz - r, px + r, pz + r), fill=color)
    start = np.array(pixel(pose[:2]))
    arrow = np.asarray((force[0], -force[1]), float) * 0.3
    if np.linalg.norm(arrow) > 75:
        arrow *= 75 / np.linalg.norm(arrow)
    if bucket_enabled and np.linalg.norm(arrow) > 2:
        end = start + arrow
        draw.line([tuple(start), tuple(end)], fill="#b44431", width=3)
        unit = arrow / np.linalg.norm(arrow)
        normal = np.array((-unit[1], unit[0]))
        draw.polygon(
            [tuple(end), tuple(end - 9 * unit + 4 * normal), tuple(end - 9 * unit - 4 * normal)],
            fill="#b44431",
        )
    draw.text(
        (35, height - 38),
        "Layer colours track initial soil height. Pale bucket outline is grid contact padding.",
        fill="#35525b",
        font=font,
    )
    draw.text((width - 130, height - 64), "x / z [m]", fill="#35525b", font=font)
    return canvas


def force_plot(rows: list[dict], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, layout="constrained")
    times = [r["time_s"] for r in rows]
    for key, label in (("force_x_n", "Fx"), ("force_z_n", "Fz")):
        axes[0].plot(times, [r[key] for r in rows], label=label)
    axes[0].set_ylabel("Bucket reaction [N]")
    axes[0].legend()
    axes[1].plot(times, [r["torque_ccw_nm"] for r in rows], color="#b44431")
    axes[1].set_ylabel("Torque CCW [N m]")
    axes[1].set_xlabel("Simulation time [s]")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def view(scene: Scene, trajectory, *, device=None, graph=True):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    from .solver import Soil

    solver = Soil(scene, device=device, graph=graph)
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(bottom=0.26)
    points = ax.scatter(
        solver.initial[0, :, 0], solver.initial[0, :, 1], s=3, c=solver.initial[0, :, 1], cmap="YlOrBr"
    )
    (bucket_line,) = ax.plot([], [], color="#23596b", linewidth=5)
    ax.set(xlim=(0, scene.width), ylim=(0, scene.height), xlabel="x [m]", ylabel="z [m]", aspect="equal")
    ax.axhline(2 * scene.dx, color="#777777")
    sliders = [
        Slider(fig.add_axes((0.24, y, 0.50, 0.025)), label, lo, hi, valinit=value)
        for y, label, lo, hi, value in [
            (0.16, "Friction angle [deg]", 0, 55, scene.material.friction_angle_deg),
            (0.12, "Cohesion [Pa]", 0, 500, scene.material.cohesion_pa),
        ]
    ]
    apply_button = Button(fig.add_axes((0.27, 0.035, 0.22, 0.045)), "Apply and reset")
    pause_button = Button(fig.add_axes((0.54, 0.035, 0.17, 0.045)), "Pause / play")
    paused = False

    def reset(_=None):
        nonlocal solver, scene
        scene = replace(
            scene,
            material=replace(scene.material, friction_angle_deg=sliders[0].val, cohesion_pa=sliders[1].val),
        )
        solver = Soil(scene, device=str(solver.device), graph=graph)

    def toggle(_=None):
        nonlocal paused
        paused = not paused

    def key(event):
        if event.key == " ":
            toggle()
        elif event.key == "r":
            reset()

    def update(_):
        if not paused:
            if solver.time >= trajectory.duration:
                solver.reset()
            pose, twist = trajectory.command(solver.time, scene.frame_dt)
            solver.step(pose, twist)
            solver.check()
        points.set_offsets(solver.x.numpy()[0])
        vertices = bucket_vertices(scene, solver.pose.numpy()[0])
        bucket_line.set_data(vertices[:, 0], vertices[:, 1])
        force = solver.wrench()[0]
        ax.set_title(
            f"{trajectory.phase(solver.time)} | {solver.time:.2f} s | Fx {force[0]:+.1f} N, Fz {force[1]:+.1f} N"
        )
        return points, bucket_line

    apply_button.on_clicked(reset)
    pause_button.on_clicked(toggle)
    fig.canvas.mpl_connect("key_press_event", key)
    fig._soil_animation = animation.FuncAnimation(
        fig, update, interval=scene.frame_dt * 1000, cache_frame_data=False
    )
    fig._soil_widgets = (sliders, apply_button, pause_button)
    plt.show()
