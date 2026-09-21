"""Mouse drawing UI. This process imports neither Torch nor Isaac."""

from __future__ import annotations

import math
import queue
import tkinter as tk

BACKGROUND = "#131b29"
CANVAS = "#192536"
REACHABLE = "#1f3048"
UNREACHABLE = "#0f151e"
BORDER = "#4c6480"
GRID = "#2b405a"
TEXT = "#f2f6ff"
MUTED = "#a9b9d0"
REQUESTED = "#62a8ff"
EXECUTED = "#57dba8"
APPROACH = "#ffba69"
TARGET = "#ff5d67"
ERROR = "#ff7b84"
GRID_STEP = 0.05


class DrawingWindow:
    """Map a physical XZ work rectangle to an aspect-preserving drawing canvas.

    X grows to the left, as the arm appears from the default Isaac camera, so the machine is to the right of the
    canvas. ``reachable`` optionally flags, as ``[x cell][z cell]`` tiling the box from its minimum corner, where the arm
    can hold the bucket angle. The rest of the box is shaded, and stroke segments that cross it are drawn red.
    The cruise speed slider ranges from 1 mm/s to ``max_speed_mm_s`` and also retimes the path being executed.
    """

    def __init__(self, outgoing, incoming, box, speed_mm_s, reachable=None, max_speed_mm_s=120.0):
        self.outgoing, self.incoming = outgoing, incoming
        self.box = box
        self.reachable = reachable
        self.speed_mm_s = speed_mm_s
        self.stroke = []
        # Unreachable samples [m] of a rejected stroke; None while the stroke is not rejected.
        self.rejected = None
        self.drawing = False
        self.last_actual = None
        self.last_phase = None
        self.root = tk.Tk()
        self.root.title("Hydraulic sketch — draw, release, follow")
        self.root.geometry("850x830")
        self.root.configure(bg=BACKGROUND)
        tk.Label(self.root, text="Draw a path", bg=BACKGROUND, fg=TEXT, font=("Segoe UI", 22, "bold")).pack(
            anchor="w", padx=28, pady=(20, 3)
        )
        tk.Label(
            self.root,
            text="Hold the mouse to draw in the lit area · release to execute",
            bg=BACKGROUND,
            fg=MUTED,
            font=("Segoe UI", 11),
        ).pack(anchor="w", padx=28)
        speed_row = tk.Frame(self.root, bg=BACKGROUND)
        tk.Label(speed_row, text="Cruise speed", bg=BACKGROUND, fg=MUTED, font=("Segoe UI", 11)).pack(
            side="left"
        )
        # Setting the value through a variable does not call ``command``, so an unrounded --speed-mm-s is kept until
        # the slider is moved.
        self.speed_var = tk.DoubleVar(self.root, value=speed_mm_s)
        self.speed_scale = tk.Scale(
            speed_row,
            from_=1,
            to=max(max_speed_mm_s, speed_mm_s),
            resolution=1,
            orient="horizontal",
            showvalue=False,
            length=320,
            variable=self.speed_var,
            command=self.set_speed,
            # Tk paints the thumb in the widget background color.
            bg=BORDER,
            activebackground=REQUESTED,
            troughcolor=CANVAS,
            highlightthickness=0,
            borderwidth=0,
            sliderrelief="flat",
            sliderlength=22,
            width=14,
        )
        self.speed_scale.pack(side="left", padx=12)
        self.speed_text = tk.StringVar(value=f"{speed_mm_s:g} mm/s")
        tk.Label(speed_row, textvariable=self.speed_text, bg=BACKGROUND, fg=TEXT, font=("Consolas", 11)).pack(
            side="left"
        )
        speed_row.pack(anchor="w", padx=28, pady=(8, 0))
        self.canvas = tk.Canvas(self.root, bg=CANVAS, highlightthickness=0, width=800, height=550)
        # Events (accepted, rejected, done) stay visible; the 20 Hz telemetry has its own line.
        self.status = tk.StringVar(value="Ready — the bucket angle is held while drawing")
        self.status_label = tk.Label(
            self.root, textvariable=self.status, bg=BACKGROUND, fg=TEXT, font=("Segoe UI", 11, "bold")
        )
        self.telemetry = tk.StringVar(value="Waiting for the simulation…")
        telemetry_label = tk.Label(
            self.root, textvariable=self.telemetry, bg=BACKGROUND, fg=MUTED, font=("Consolas", 9)
        )
        valve_panel = tk.Frame(self.root, bg=BACKGROUND)
        tk.Label(
            valve_panel,
            text="Applied valve commands  (−1 … +1)",
            bg=BACKGROUND,
            fg="#8da5c1",
            font=("Segoe UI", 9),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 2))
        self.valve_meters = []
        for column, name in enumerate(("Boom", "Arm", "Bucket")):
            cell = tk.Frame(valve_panel, bg=BACKGROUND)
            cell.grid(row=1, column=column, sticky="ew", padx=(0, 16 if column < 2 else 0))
            valve_panel.grid_columnconfigure(column, weight=1)
            value = tk.StringVar(value="+0.00")
            tk.Label(cell, text=name, bg=BACKGROUND, fg=MUTED, font=("Segoe UI", 9)).pack(side="left")
            tk.Label(cell, textvariable=value, bg=BACKGROUND, fg=TEXT, font=("Consolas", 9)).pack(
                side="right"
            )
            meter = tk.Canvas(cell, height=18, bg=CANVAS, highlightthickness=0)
            meter.pack(fill="x", expand=True, padx=(8, 0), pady=2)
            self.valve_meters.append((meter, value))
        legend = tk.Frame(self.root, bg=BACKGROUND)
        for text, color, symbol in (
            ("Requested", REQUESTED, "━"),
            ("Executed", EXECUTED, "━"),
            ("Approach", APPROACH, "━"),
            ("Live target", TARGET, "●"),
        ):
            tk.Label(legend, text=symbol + " " + text, fg=color, bg=BACKGROUND, font=("Segoe UI", 11)).pack(
                side="left", padx=(0, 24)
            )
        if reachable is not None:
            # A lit patch with a dark corner, like the map itself.
            swatch = tk.Canvas(legend, width=18, height=14, bg=BACKGROUND, highlightthickness=0)
            swatch.create_rectangle(0, 0, 18, 14, fill=REACHABLE, outline="")
            swatch.create_polygon(18, 0, 4, 0, 18, 11, fill=UNREACHABLE, outline="")
            swatch.pack(side="left", padx=(0, 6))
            tk.Label(legend, text="Dark: out of reach", fg=MUTED, bg=BACKGROUND, font=("Segoe UI", 11)).pack(
                side="left"
            )
        tk.Button(
            legend, text="Stop", command=self.stop, bg="#31445f", fg="white", relief="flat", padx=16
        ).pack(side="right")
        legend.pack(side="bottom", fill="x", padx=28, pady=(0, 16))
        valve_panel.pack(side="bottom", fill="x", padx=28, pady=(0, 10))
        telemetry_label.pack(side="bottom", anchor="w", padx=28, pady=(0, 6))
        self.status_label.pack(side="bottom", anchor="w", padx=28)
        # Packed last, so the canvas rather than the controls below it shrinks in a short window.
        self.canvas.pack(fill="both", expand=True, padx=24, pady=(14, 8))
        self.canvas.bind("<Configure>", self.resize)
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.motion)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.root.bind("<Escape>", lambda event: self.stop())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.bounds = (50, 40, 750, 520)
        self.root.after(30, self.poll)

    def set_status(self, text: str, error: bool = False):
        """Show an event message until the next event replaces it."""
        self.status.set(text)
        self.status_label.configure(fg=ERROR if error else TEXT)

    def resize(self, event):
        dx, dz = self.box[1] - self.box[0], self.box[3] - self.box[2]
        scale = min((event.width - 90) / dx, (event.height - 60) / dz)
        width, height = scale * dx, scale * dz
        left, top = (event.width - width) / 2, (event.height - height) / 2
        self.bounds = (left, top, left + width, top + height)
        self.canvas.delete("grid")
        self.canvas.create_rectangle(
            *self.bounds, fill=CANVAS if self.reachable is None else REACHABLE, outline="", tags="grid"
        )
        self.shade_unreachable()
        # Grid lines at round 50 mm coordinates, not at fractions of the box.
        for k in range(math.ceil(self.box[0] / GRID_STEP), math.floor(self.box[1] / GRID_STEP) + 1):
            x = self.to_canvas((k * GRID_STEP, self.box[2]))[0]
            if left + 1 < x < left + width - 1:
                self.canvas.create_line(x, top, x, top + height, fill=GRID, tags="grid")
        for k in range(math.ceil(self.box[2] / GRID_STEP), math.floor(self.box[3] / GRID_STEP) + 1):
            y = self.to_canvas((self.box[0], k * GRID_STEP))[1]
            if top + 1 < y < top + height - 1:
                self.canvas.create_line(left, y, left + width, y, fill=GRID, tags="grid")
        self.canvas.create_rectangle(*self.bounds, outline=BORDER, width=2, tags="grid")
        self.canvas.create_text(
            left,
            top + height + 20,
            anchor="w",
            fill="#8da5c1",
            tags="grid",
            text=f"X {1000 * self.box[1]:.0f} → {1000 * self.box[0]:.0f} mm, machine to the right · "
            f"grid {1000 * GRID_STEP:.0f} mm",
        )
        self.canvas.create_text(
            left,
            top - 15,
            anchor="w",
            fill="#8da5c1",
            tags="grid",
            text=f"Z {1000 * self.box[2]:.0f} → {1000 * self.box[3]:.0f} mm",
        )
        self.canvas.tag_lower("grid")
        # Existing traces are physical points and are redrawn correctly after a resize.
        self.redraw_requested()
        self.canvas.delete("actual")
        self.last_actual = None

    def shade_unreachable(self):
        """Darken unreachable cells, merged into vertical runs to keep the item count low."""
        if self.reachable is None:
            return
        columns, rows = len(self.reachable), len(self.reachable[0])

        def corner(i, j):
            return self.to_canvas(
                (
                    self.box[0] + i * (self.box[1] - self.box[0]) / columns,
                    self.box[2] + j * (self.box[3] - self.box[2]) / rows,
                )
            )

        for i, column in enumerate(self.reachable):
            j = 0
            while j < rows:
                if column[j]:
                    j += 1
                    continue
                start = j
                while j < rows and not column[j]:
                    j += 1
                self.canvas.create_rectangle(
                    *corner(i, start), *corner(i + 1, j), fill=UNREACHABLE, outline=UNREACHABLE, tags="grid"
                )

    def is_reachable(self, point) -> bool:
        """Look up the reachability cell of a point [m]; without a map everything counts as reachable."""
        if self.reachable is None:
            return True
        columns, rows = len(self.reachable), len(self.reachable[0])
        i = int((point[0] - self.box[0]) / (self.box[1] - self.box[0]) * columns)
        j = int((point[1] - self.box[2]) / (self.box[3] - self.box[2]) * rows)
        return bool(self.reachable[min(max(i, 0), columns - 1)][min(max(j, 0), rows - 1)])

    def redraw_requested(self):
        """Draw the stroke (red once rejected) and mark where a rejected stroke is out of reach."""
        self.canvas.delete("requested", "rejected")
        if len(self.stroke) > 1:
            self.canvas.create_line(
                *[value for point in self.stroke for value in self.to_canvas(point)],
                fill=REQUESTED if self.rejected is None else ERROR,
                width=2,
                tags="requested",
            )
        for point in self.rejected or ():
            x, y = self.to_canvas(point)
            for sign in (-1, 1):
                self.canvas.create_line(x - 4, y - 4 * sign, x + 4, y + 4 * sign, fill=ERROR, tags="rejected")

    def update_valves(self, valves):
        """Show applied post-tanh valve commands on a centered -1..1 scale."""
        for raw, (meter, label) in zip(valves, self.valve_meters):
            value = max(-1.0, min(1.0, float(raw)))
            width = max(20, meter.winfo_width())
            height = max(10, meter.winfo_height())
            center = width / 2
            endpoint = center + value * (width / 2 - 3)
            meter.delete("all")
            meter.create_line(center, 2, center, height - 2, fill=BORDER)
            meter.create_rectangle(
                min(center, endpoint),
                4,
                max(center, endpoint),
                height - 4,
                fill=REQUESTED if value >= 0 else APPROACH,
                outline="",
            )
            label.set(f"{value:+.2f}")

    def to_world(self, x, y):
        left, top, right, bottom = self.bounds
        x, y = min(max(x, left), right), min(max(y, top), bottom)
        return [
            self.box[1] - (x - left) / (right - left) * (self.box[1] - self.box[0]),
            self.box[3] - (y - top) / (bottom - top) * (self.box[3] - self.box[2]),
        ]

    def to_canvas(self, point):
        left, top, right, bottom = self.bounds
        return (
            left + (self.box[1] - point[0]) / (self.box[1] - self.box[0]) * (right - left),
            top + (self.box[3] - point[1]) / (self.box[3] - self.box[2]) * (bottom - top),
        )

    def press(self, event):
        self.drawing = True
        self.stroke = [self.to_world(event.x, event.y)]
        self.rejected = None
        self.canvas.delete("requested", "rejected", "actual")
        self.last_actual = self.last_phase = None
        self.outgoing.put({"kind": "cancel", "clear": True})
        if self.is_reachable(self.stroke[0]):
            self.set_status("Drawing — release to execute")
        else:
            self.set_status("Started outside the reachable area — this stroke will be rejected", error=True)

    def motion(self, event):
        if not self.drawing:
            return
        point = self.to_world(event.x, event.y)
        if sum((a - b) ** 2 for a, b in zip(point, self.stroke[-1])) < 1e-8:
            return
        inside = self.is_reachable(point)
        if not inside and self.is_reachable(self.stroke[-1]):
            self.set_status("Crossed into the shaded area — this stroke will be rejected", error=True)
        self.canvas.create_line(
            *self.to_canvas(self.stroke[-1]),
            *self.to_canvas(point),
            fill=REQUESTED if inside and self.is_reachable(self.stroke[-1]) else ERROR,
            width=2,
            tags="requested",
        )
        self.stroke.append(point)

    def release(self, event):
        if self.drawing:
            self.motion(event)
            self.drawing = False
            self.outgoing.put({"kind": "stroke", "points": self.stroke})
            self.set_status("Validating path…")

    def set_speed(self, value):
        """Send a moved slider's cruise speed [mm/s]; it applies to the path being executed too."""
        speed = float(value)
        if speed == self.speed_mm_s:
            return
        self.speed_mm_s = speed
        self.speed_text.set(f"{speed:g} mm/s")
        self.outgoing.put({"kind": "speed", "mm_s": speed})

    def stop(self):
        self.drawing = False
        self.last_phase = None
        self.outgoing.put({"kind": "cancel"})
        self.set_status("Stopped — holding position")

    def close(self):
        self.outgoing.put({"kind": "close"})
        self.root.destroy()

    def show_state(self, message):
        """Draw the measured tip, live target and executed trace; update telemetry and valve meters."""
        point = self.to_canvas(message["position"])
        target = self.to_canvas(message["target"])
        phase = message["phase"]
        self.update_valves(message["valves"])
        if not self.drawing:
            if self.last_actual is not None and phase in ("approach", "drawing") and phase == self.last_phase:
                self.canvas.create_line(
                    *self.last_actual,
                    *point,
                    width=2,
                    tags="actual",
                    fill=APPROACH if phase == "approach" else EXECUTED,
                )
            if self.last_phase == "drawing" and phase == "hold":
                self.set_status("Done — draw another path")
            self.last_actual = point
            self.last_phase = phase
        self.telemetry.set(
            f"{phase.capitalize()} · tracking error {message['error_mm']:.1f} mm · "
            f"speed {message['speed_mm_s']:.1f} mm/s"
        )
        self.canvas.delete("tip", "target")
        for (x, y), color, tag in ((point, "#ffffff", "tip"), (target, TARGET, "target")):
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="", tags=tag)

    def poll(self):
        try:
            while True:
                message = self.incoming.get_nowait()
                if message["kind"] == "close":
                    self.root.destroy()
                    return
                if message["kind"] == "error":
                    self.rejected = message.get("points", [])
                    self.redraw_requested()
                    self.set_status(message["text"], error=True)
                elif message["kind"] == "path":
                    self.stroke = message["points"]
                    self.redraw_requested()
                    self.set_status("Path accepted — executing")
                else:
                    self.show_state(message)
        except queue.Empty:
            pass
        self.root.after(30, self.poll)


def open_drawing_window(outgoing, incoming, box, speed_mm_s, reachable=None, max_speed_mm_s=120.0):
    """Run the visible drawing window in a spawned process."""
    DrawingWindow(outgoing, incoming, box, speed_mm_s, reachable, max_speed_mm_s).root.mainloop()
