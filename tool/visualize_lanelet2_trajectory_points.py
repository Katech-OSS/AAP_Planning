#!/usr/bin/env python3
"""
Plot Lanelet2 OSM map and all trajectory points (scatter colored by velocity).

Differences from the line/animation version:
  - No line plot of ego path.
  - No file outputs (no PNG/MP4/GIF); just shows a matplotlib window.
  - Supports JSON (single message) and JSONL (one message per line) trajectory dumps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer

# User-adjustable manual translation for the lanelet map (meters), tuned on trajectory2
MANUAL_DX = -50.4
MANUAL_DY = 7.2
# Fixed reference start position (world) to anchor map and trajectories
REF_START_X = 3708.456298828125
REF_START_Y = 73666.421875
# Fixed start/goal markers (world frame)
START: Tuple[float, float] = (3733.0, 73679.0)
GOALS: Dict[int, Tuple[float, float]] = {
    1: (3831.229, 73730.367),
    2: (3770.879, 73729.656),
    3: (3831.229, 73730.367),
}
# Fixed plot bounds per scenario, derived from Scenario_{N}_1.jsonl after anchoring to REF_START
SCENARIO_FIXED_BOUNDS: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    1: ((3703.456298828125, 3812.3225495931997), (73661.421875, 73722.95039384908)),
    2: ((3703.456298828125, 3762.1529604873735), (73661.421875, 73722.23938998018)),
    3: ((3703.456298828125, 3812.3225495931997), (73661.421875, 73722.95039382219)),
}


# ---------------------------
# Coordinate transform helpers
# ---------------------------
def build_enu_transformer(origin_lat: float, origin_lon: float) -> Transformer:
    """Azimuthal equidistant projection centered at origin to get local ENU meters."""
    proj_str = (
        f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)


def latlon_to_local_xy(
    lat: float, lon: float, transformer: Transformer
) -> Tuple[float, float]:
    """Convert latitude/longitude to local ENU meters (east, north)."""
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


# ---------------------------
# Lanelet2 OSM parsing
# ---------------------------
def load_lanelet_map(
    osm_path: Path, origin_lat: Optional[float], origin_lon: Optional[float]
) -> Tuple[List[np.ndarray], float, float]:
    """
    Parse a Lanelet2-format OSM file and convert all way node sequences to local ENU coords.

    Returns:
        polylines: list of (N,2) arrays representing way geometries
        origin_lat, origin_lon: origin actually used
    """
    tree = ET.parse(osm_path)
    root = tree.getroot()

    node_coords: Dict[str, Tuple[float, float]] = {}
    for node in root.findall("node"):
        node_id = node.attrib["id"]
        lat = float(node.attrib["lat"])
        lon = float(node.attrib["lon"])
        node_coords[node_id] = (lat, lon)

    if origin_lat is None or origin_lon is None:
        first_lat, first_lon = next(iter(node_coords.values()))
        origin_lat = first_lat if origin_lat is None else origin_lat
        origin_lon = first_lon if origin_lon is None else origin_lon

    transformer = build_enu_transformer(origin_lat, origin_lon)

    node_xy: Dict[str, Tuple[float, float]] = {}
    for node_id, (lat, lon) in node_coords.items():
        node_xy[node_id] = latlon_to_local_xy(lat, lon, transformer)

    polylines: List[np.ndarray] = []
    for way in root.findall("way"):
        coords: List[Tuple[float, float]] = []
        for nd in way.findall("nd"):
            ref = nd.attrib["ref"]
            if ref in node_xy:
                coords.append(node_xy[ref])
        if len(coords) >= 2:
            polylines.append(np.array(coords))

    return polylines, origin_lat, origin_lon


# ---------------------------
# Trajectory parsing
# ---------------------------
def _extract_points_from_message(msg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract x, y, velocity, time_from_start from a trajectory message dict.
    message shape follows Autoware /planning/scenario_planning/trajectory.
    """
    if "message" in msg:
        msg = msg["message"]

    pts = msg.get("points", [])
    xs: List[float] = []
    ys: List[float] = []
    vels: List[float] = []
    times: List[float] = []
    for p in pts:
        pose = p.get("pose", {})
        pos = pose.get("position", {})
        xs.append(float(pos.get("x", 0.0)))
        ys.append(float(pos.get("y", 0.0)))
        vels.append(float(p.get("longitudinal_velocity_mps", 0.0)))
        t_sec = float(p.get("time_from_start", {}).get("sec", 0.0))
        t_nsec = float(p.get("time_from_start", {}).get("nanosec", 0.0))
        times.append(t_sec + t_nsec * 1e-9)

    return np.array(xs), np.array(ys), np.array(vels), np.array(times)


def _extract_timestamp_sec(msg: Dict) -> Optional[float]:
    """Return message timestamp in seconds (including nanoseconds) if available."""
    src = msg.get("message", msg)
    stamp = src.get("header", {}).get("stamp")
    if isinstance(stamp, dict):
        sec = float(stamp.get("sec", 0.0))
        nanosec = float(stamp.get("nanosec", 0.0))
        return sec + nanosec * 1e-9

    ts_raw = msg.get("timestamp")
    if ts_raw is not None:
        try:
            ts = float(ts_raw)
        except (TypeError, ValueError):
            return None
        # Heuristic: treat very large values as nanoseconds
        if ts > 1e12:
            ts *= 1e-9
        return ts

    return None


def load_trajectory_file(
    path: Path, t_max: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]], float]:
    """
    Load trajectory messages from JSON or JSONL.
    Returns:
        all_x, all_y, all_vel_kmh, all_time_from_start (concatenated for bounds/color scaling)
        messages: list of dicts with keys xs, ys, vels (km/h), times, timestamp_sec, delta_traj_sec, freq_traj_Hz.
        freq_traj_Hz_mean: mean frequency over the file (ignores missing/zero deltas).
    Optional t_max filters points with time_from_start <= t_max (per message).
    """
    xs_all: List[float] = []
    ys_all: List[float] = []
    vels_all_kmh: List[float] = []
    times_all: List[float] = []
    messages: List[Dict[str, object]] = []

    def _process_message(msg: Dict) -> None:
        timestamp_sec = _extract_timestamp_sec(msg)
        xs, ys, vels, times = _extract_points_from_message(msg)
        if t_max is not None:
            mask = times <= t_max
            xs, ys, vels, times = xs[mask], ys[mask], vels[mask], times[mask]
        if xs.size == 0:
            return
        vels_kmh = vels * 3.6
        xs_all.extend(xs.tolist())
        ys_all.extend(ys.tolist())
        vels_all_kmh.extend(vels_kmh.tolist())
        times_all.extend(times.tolist())
        messages.append(
            {
                "xs": xs,
                "ys": ys,
                "vels": vels_kmh,
                "times": times,
                "timestamp_sec": timestamp_sec,
            }
        )

    if path.suffix.lower() == ".jsonl":
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                msg = json.loads(line)
                _process_message(msg)
    else:
        msg = json.loads(path.read_text())
        _process_message(msg)

    if not xs_all:
        raise ValueError("No trajectory points loaded from the provided file.")

    # Compute inter-message deltas and frequencies
    freq_values: List[float] = []
    prev_ts: Optional[float] = None
    for idx, msg in enumerate(messages):
        ts = msg.get("timestamp_sec")
        if ts is None or prev_ts is None:
            msg["delta_traj_sec"] = 0.0 if idx == 0 else None
            msg["freq_traj_Hz"] = None
        else:
            delta = ts - prev_ts
            msg["delta_traj_sec"] = delta
            if delta > 0.0:
                freq = 1.0 / delta
                msg["freq_traj_Hz"] = freq
                freq_values.append(freq)
            else:
                msg["freq_traj_Hz"] = None
        if ts is not None:
            prev_ts = ts

    freq_traj_Hz_mean = float(np.mean(freq_values)) if freq_values else 0.0

    return (
        np.array(xs_all),
        np.array(ys_all),
        np.array(vels_all_kmh),
        np.array(times_all),
        messages,
        freq_traj_Hz_mean,
    )


# ---------------------------
# Plotting
# ---------------------------
def plot_lanelet_map(ax: plt.Axes, polylines: Iterable[np.ndarray]) -> None:
    """Draw lanelet polylines."""
    line_segments = [pline[:, :2] for pline in polylines]
    lc = LineCollection(line_segments, colors="0.7", linewidths=0.8, alpha=0.8)
    ax.add_collection(lc)


def plot_trajectory_points(
    ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, vels: np.ndarray
) -> plt.Collection:
    """Scatter all trajectory points colored by longitudinal velocity."""
    sc = ax.scatter(xs, ys, c=vels, s=8, cmap="RdYlGn", alpha=0.9)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = ax.figure.colorbar(sc, cax=cax)
    cbar.set_label("Velocity [km/h]")
    cbar.ax.tick_params(labelsize=8)
    return sc


def compute_bounds(
    polylines: Iterable[np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    extra_points: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[float, float, float, float]:
    """
    Compute bounding box including trajectory points and the fixed reference start,
    then apply a margin. Keeps the reference location visible without retuning offsets.
    """
    base_pts_x = [REF_START_X]
    base_pts_y = [REF_START_Y]
    if extra_points:
        for x_pt, y_pt in extra_points:
            base_pts_x.append(x_pt)
            base_pts_y.append(y_pt)
    all_x = np.concatenate([xs, np.array(base_pts_x)])
    all_y = np.concatenate([ys, np.array(base_pts_y)])
    xmin, xmax = float(all_x.min()), float(all_x.max())
    ymin, ymax = float(all_y.min()), float(all_y.max())
    dx = xmax - xmin
    dy = ymax - ymin
    margin_x = max(5.0, 0.05 * dx)
    margin_y = max(5.0, 0.05 * dy)
    return xmin - margin_x, xmax + margin_x, ymin - margin_y, ymax + margin_y


def translate_polylines(polylines: List[np.ndarray], dx: float, dy: float) -> List[np.ndarray]:
    """Translate all polylines by (dx, dy)."""
    return [pline + np.array([dx, dy]) for pline in polylines]


def polylines_mean(polylines: List[np.ndarray]) -> Tuple[float, float]:
    """Compute mean x,y over all polyline points."""
    xs: List[float] = []
    ys: List[float] = []
    for pl in polylines:
        if pl.size == 0:
            continue
        xs.extend(pl[:, 0].tolist())
        ys.extend(pl[:, 1].tolist())
    return float(np.mean(xs)), float(np.mean(ys))


# ---------------------------
# Vehicle drawing helpers
# ---------------------------
def create_vehicle_shape(length: float = 6.0, width: float = 2.5) -> np.ndarray:
    """
    Build a simple vehicle silhouette (non-rectangular) in the local vehicle frame.

    Local frame: x forward, y left, origin = rear center (anchor).
    Shape uses a pointed nose and inset shoulders while keeping the rear center at (0, 0).
    """
    half_w = width * 0.5
    nose_x = length
    shoulder_x = length * 0.8
    mid_x = length * 0.3
    rear_y = width * 0.35  # slight taper near rear
    return np.array(
        [
            [nose_x, 0.0],  # nose
            [shoulder_x, half_w],
            [mid_x, half_w],
            [0.0, rear_y],
            [0.0, -rear_y],
            [mid_x, -half_w],
            [shoulder_x, -half_w],
        ]
    )


def update_vehicle_pose(
    patch: Polygon, base_shape: np.ndarray, anchor_x: float, anchor_y: float, yaw: float
) -> None:
    """
    Rotate the base shape about the rear-center anchor and translate to (anchor_x, anchor_y).
    """
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    rotated = base_shape @ rot.T
    translated = rotated + np.array([anchor_x, anchor_y])
    patch.set_xy(translated)


def compute_vehicle_yaw(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Compute vehicle heading using Method A (trajectory point difference).

    Uses the direction from point 1 to point 10 (1-based). Falls back to the last
    available point if fewer than 10 points exist.
    """
    if xs.size == 0 or ys.size == 0:
        return 0.0
    target_idx = min(9, xs.size - 1)  # 9 == 10th point (0-based)
    dx = float(xs[target_idx] - xs[0])
    dy = float(ys[target_idx] - ys[0])
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return float(np.arctan2(dy, dx))


def create_animation(
    fig: plt.Figure,
    ax: plt.Axes,
    polylines: Iterable[np.ndarray],
    all_xs: np.ndarray,
    all_ys: np.ndarray,
    all_vels_kmh: np.ndarray,
    messages: List[Dict[str, object]],
    max_frames: int,
    frame_step: int,
    freq_traj_Hz_mean: float,
    start_point_plot: Tuple[float, float],
    goal_point_plot: Tuple[float, float],
    scenario: int,
    file_label: str,
) -> animation.FuncAnimation:
    """
    Create an animation where each frame shows only the current message's trajectory points.
    Previous frames are cleared by updating the scatter offsets to the current message only.
    """
    if not messages:
        raise ValueError("No trajectory messages to animate.")

    # Compute color scaling
    vmin = float(all_vels_kmh.min())
    vmax = float(all_vels_kmh.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    # Map (draw once, stays on axes)
    plot_lanelet_map(ax, polylines)

    # Fixed start/goal markers (draw once, reused across frames)
    start_x_plot, start_y_plot = start_point_plot
    goal_x_plot, goal_y_plot = goal_point_plot
    start_scatter = ax.scatter(
        [start_x_plot],
        [start_y_plot],
        marker="s",
        c="magenta",
        alpha=0.5,
        s=80,
        label="Initial",
    )
    goal_scatter = ax.scatter(
        [goal_x_plot],
        [goal_y_plot],
        marker="*",
        c="magenta",
        alpha=0.5,
        s=140,
        label="Goal",
    )

    # Base scatter (empty to start); set clim so colorbar matches full range
    sc = ax.scatter([], [], c=[], s=8, cmap="RdYlGn", alpha=0.9, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Velocity [km/h]")
    cbar.ax.tick_params(labelsize=8)

    fixed_bounds = SCENARIO_FIXED_BOUNDS.get(scenario)
    if fixed_bounds is not None:
        (xmin, xmax), (ymin, ymax) = fixed_bounds
    else:
        xmin, xmax, ymin, ymax = compute_bounds(
            polylines, all_xs, all_ys, extra_points=[(start_x_plot, start_y_plot), (goal_x_plot, goal_y_plot)]
        )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title(f"[{file_label} Result]", fontsize=12, fontweight="bold")
    ax.legend(handles=[start_scatter, goal_scatter], loc="lower right")

    # Frame indices: subsample messages (not points)
    if frame_step > 1:
        frame_indices = np.arange(0, len(messages), frame_step, dtype=int)
    else:
        frame_indices = np.arange(len(messages))
    if max_frames and len(frame_indices) > max_frames:
        frame_indices = np.linspace(0, len(messages) - 1, num=max_frames, dtype=int)

    vehicle_shape_local = create_vehicle_shape(length=6.0, width=2.5)
    vehicle_patch = Polygon(
        vehicle_shape_local,
        closed=True,
        color="black",
        alpha=0.4,
        zorder=3,
    )
    ax.add_patch(vehicle_patch)

    def _format_freq(freq_val: Optional[float]) -> str:
        if freq_val is None:
            freq_val = freq_traj_Hz_mean if freq_traj_Hz_mean > 0 else None
        if freq_val is None:
            return f"{'N/A':>7} [Hz]"
        return f"{freq_val:7.2f} [Hz]"

    freq_text = ax.text(
        0.02,
        0.95,
        _format_freq(messages[0].get("freq_traj_Hz")),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        fontfamily="monospace",  # fixed-width to keep [Hz] aligned
        color="black",
        zorder=4,
    )

    def init():
        sc.set_offsets(np.empty((0, 2)))
        sc.set_array(np.array([]))
        first_msg = messages[frame_indices[0]]
        vx = float(first_msg["xs"][0])
        vy = float(first_msg["ys"][0])
        vyaw = compute_vehicle_yaw(first_msg["xs"], first_msg["ys"])
        update_vehicle_pose(vehicle_patch, vehicle_shape_local, vx, vy, vyaw)
        freq_text.set_text(_format_freq(first_msg.get("freq_traj_Hz")))
        return (sc, vehicle_patch, freq_text)

    def animate(i: int):
        idx = frame_indices[i]
        msg = messages[idx]
        xs_m = msg["xs"]
        ys_m = msg["ys"]
        vels_m = msg["vels"]
        pts = np.column_stack((xs_m, ys_m))
        sc.set_offsets(pts)
        sc.set_array(vels_m)
        cx = float(xs_m[0])
        cy = float(ys_m[0])
        yaw = compute_vehicle_yaw(xs_m, ys_m)
        update_vehicle_pose(vehicle_patch, vehicle_shape_local, cx, cy, yaw)
        freq_text.set_text(_format_freq(msg.get("freq_traj_Hz")))
        return (sc, vehicle_patch, freq_text)

    # Frame interval: use a fixed reasonable value (50 ms) to keep export predictable
    interval_ms = 50.0

    return animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(frame_indices),
        interval=interval_ms,
        blit=False,  # keep lanelet map and axes intact on every frame
        repeat=False,
        save_count=len(frame_indices),
    )


# ---------------------------
# CLI
# ---------------------------
def _parse_scenario(value: str) -> int:
    """Validate scenario argument and map to an int key."""
    try:
        scenario = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scenario must be an integer (1, 2, or 3).") from exc
    if scenario not in GOALS:
        raise argparse.ArgumentTypeError(f"scenario must be one of {sorted(GOALS.keys())}.")
    return scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Lanelet2 map with trajectory points (scatter).")
    parser.add_argument("--lanelet-osm", required=True, type=Path, help="Path to lanelet2 OSM file.")
    parser.add_argument("--trajectory-file", required=True, type=Path, help="Path to trajectory JSON/JSONL file.")
    parser.add_argument("--origin-lat", type=float, default=None, help="Origin latitude for ENU frame.")
    parser.add_argument("--origin-lon", type=float, default=None, help="Origin longitude for ENU frame.")
    parser.add_argument("--dx", type=float, default=0.0, help="Offset (meters) to add to trajectory x.")
    parser.add_argument("--dy", type=float, default=0.0, help="Offset (meters) to add to trajectory y.")
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Optional time_from_start upper bound (seconds) to filter points per message.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the matplotlib window (useful when only saving GIF).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="Maximum number of frames to save in the GIF (subsamples points if needed).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Use every Nth point as a frame to reduce output size/time (>=1).",
    )
    parser.add_argument(
        "--scenario",
        type=_parse_scenario,
        required=True,
        help="Scenario number selecting the goal location (1, 2, or 3).",
    )
    return parser.parse_args()


def main() -> None:
    plt.ioff()  # Disable interactive mode to prevent popup windows during batch runs
    args = parse_args()

    polylines, origin_lat, origin_lon = load_lanelet_map(
        args.lanelet_osm, args.origin_lat, args.origin_lon
    )
    # print(f"Using ENU origin lat={origin_lat:.8f}, lon={origin_lon:.8f}")

    xs, ys, vels_kmh, times, messages, freq_traj_Hz_mean = load_trajectory_file(
        args.trajectory_file, t_max=args.t_max
    )
    if xs.size == 0:
        raise ValueError("No trajectory points loaded.")

    # Align trajectory start to fixed reference so any file lands on same road spot.
    traj_start_x = float(xs[0])
    traj_start_y = float(ys[0])
    traj_shift_x = REF_START_X - traj_start_x
    traj_shift_y = REF_START_Y - traj_start_y
    xs = xs + traj_shift_x + args.dx
    ys = ys + traj_shift_y + args.dy
    for m in messages:
        m["xs"] = m["xs"] + traj_shift_x + args.dx
        m["ys"] = m["ys"] + traj_shift_y + args.dy

    # ---------------- Translation of lanelet map to fixed world frame ----------------
    # Anchor map to reference start, then apply manual tweak (trajectory-independent)
    map_mean_x, map_mean_y = polylines_mean(polylines)
    base_dx = REF_START_X - map_mean_x
    base_dy = REF_START_Y - map_mean_y
    final_dx = base_dx + MANUAL_DX
    final_dy = base_dy + MANUAL_DY

    # Apply translation to map
    polylines = translate_polylines(polylines, final_dx, final_dy)

    # print(
    #     f"Map translation applied: base_to_ref=({base_dx:.3f}, {base_dy:.3f}), "
    #     f"manual=({MANUAL_DX:.3f}, {MANUAL_DY:.3f}), final=({final_dx:.3f}, {final_dy:.3f}), "
    #     f"traj_shift=({traj_shift_x:.3f}, {traj_shift_y:.3f}), ref_start=({REF_START_X:.3f}, {REF_START_Y:.3f})"
    # )

    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute marker positions so they live in the same frame as the shifted trajectory
    start_x_raw, start_y_raw = START
    goal_x_raw, goal_y_raw = GOALS[args.scenario]
    start_x_shifted = start_x_raw + traj_shift_x + args.dx
    start_y_shifted = start_y_raw + traj_shift_y + args.dy
    goal_x_shifted = goal_x_raw + traj_shift_x + args.dx
    goal_y_shifted = goal_y_raw + traj_shift_y + args.dy
    start_point_plot = (start_x_shifted, start_y_shifted)
    goal_point_plot = (goal_x_shifted, goal_y_shifted)
    file_label = args.trajectory_file.name

    ani = create_animation(
        fig,
        ax,
        polylines,
        xs,
        ys,
        vels_kmh,
        messages,
        max_frames=max(1, args.max_frames),
        frame_step=max(1, args.frame_step),
        freq_traj_Hz_mean=freq_traj_Hz_mean,
        start_point_plot=start_point_plot,
        goal_point_plot=goal_point_plot,
        scenario=args.scenario,
        file_label=file_label,
    )

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{args.trajectory_file.stem}_visual.gif"
    # Higher fps and dpi for smoother GIF output
    writer = animation.PillowWriter(fps=20)
    ani.save(gif_path, writer=writer, dpi=200)
    print(f"Saved GIF animation to {gif_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
