"""Temporal multi-object tracker for 3D detections (no GUI dependencies).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from yaw_stats import circular_mean_deg, circular_std_deg


@dataclass
class _Track:
    """Sliding-window history of one tracked object.
    """

    track_id: int
    class_name: str
    centers: deque = field(default_factory=deque)
    extents: deque = field(default_factory=deque)
    yaws_deg: deque = field(default_factory=deque)
    confidences: deque = field(default_factory=deque)
    yaw_reliables: deque = field(default_factory=deque)
    misses: int = 0
    stable_latched: bool = False
    enter_streak: int = 0

    def add(self, detection: Any, window_size: int) -> None:
        """Append one detection, drop anything older than `window_size`."""
        self.centers.append(np.asarray(detection.center_xyz, dtype=np.float64))
        self.extents.append(np.asarray(detection.extent_xyz, dtype=np.float64))
        self.yaws_deg.append(float(detection.yaw_deg))
        self.confidences.append(float(detection.confidence))
        self.yaw_reliables.append(bool(getattr(detection, "yaw_reliable", False)))
        self.misses = 0
        while len(self.centers) > window_size:
            self.centers.popleft()
            self.extents.popleft()
            self.yaws_deg.popleft()
            self.confidences.popleft()
            self.yaw_reliables.popleft()

    @property
    def hits(self) -> int:
        return len(self.centers)

    def filtered_center(self) -> np.ndarray:
        return np.median(np.stack(self.centers), axis=0)

    def filtered_extent(self) -> np.ndarray:
        return np.median(np.stack(self.extents), axis=0)

    def filtered_yaw_deg(self) -> float:
        return circular_mean_deg(list(self.yaws_deg))

    def position_std_m(self) -> float:
        if len(self.centers) < 2:
            return 0.0
        per_axis_std = np.std(np.stack(self.centers), axis=0)
        return float(np.linalg.norm(per_axis_std))

    def yaw_std_deg(self) -> float:
        return circular_std_deg(list(self.yaws_deg))

    def confidence_mean(self) -> float:
        return float(np.mean(self.confidences))

    def yaw_reliable(self) -> bool:
        """Track-level yaw reliability: majority of the window has an elongated footprint."""
        if not self.yaw_reliables:
            return False
        return float(np.mean(self.yaw_reliables)) >= 0.5


@dataclass
class TrackedDetection:
    """Per frame detection enriched with temporal filtering, the tracker output.

    Wraps the raw `Detection3D` from pipeline.py and tags it with a track_id: 
    each track_id has a history of detections, and is characterized by:
    hits, confidence_mean, filtered center/extent/yaw, position_std_m,
    yaw_std_deg, and the `stable` flag.
    """

    raw: Any
    track_id: int
    hits: int
    confidence_mean: float
    center_filtered_xyz: list[float]
    extent_filtered_xyz: list[float]
    yaw_filtered_deg: float
    yaw_filtered_rad: float
    position_std_m: float
    yaw_std_deg: float
    yaw_reliable: bool
    stable: bool

    def __getattr__(self, name: str) -> Any:
        if name == "raw":
            raise AttributeError(name)
        return getattr(self.raw, name)


@dataclass
class TrackerConfig:
    """Default thresholds for the tracker (Overwritten by config/sense.yaml)."""

    window_size: int = 10          # frames kept per track 
    min_hits: int = 6              # hits required for considered stable
    max_misses: int = 5           # frames without a hit before a track is dropped
    max_position_std_m: float = 0.008   # How much the center can move per frame to be considered stable
    max_yaw_std_deg: float = 10.0
    min_confidence: float = 0.8
    assoc_max_dist_m: float = 0.04      # association gate (4 cm)
    stable_enter_frames: int = 3   # consecutive frames meeting strict criteria before latching stable
    stable_hysteresis: float = 1.5      # exit thresholds = enter thresholds * this (looser to leave)
    stable_conf_margin: float = 0.05    # exit when mean confidence drops below (min_confidence - margin)


class DetectionTracker:
    """Associates per frame detections to persistent tracks and filters them.

    Call `update(detections_3d)` once per frame. Returns one TrackedDetection per
    currently visible detection.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: list[_Track] = []
        self._next_id = 0

    def update(self, detections_3d: list[Any]) -> list[TrackedDetection]:
        cfg = self.config

        # Only detections with a valid 3D center can be tracked / localized.
        localizable = [
            (idx, det)
            for idx, det in enumerate(detections_3d)
            if getattr(det, "center_xyz", None) is not None
        ]

        matched_track_for_idx: dict[int, _Track] = {}
        used_tracks: set[int] = set()

        # Greedy nearest-neighbour association within the gate, per class.
        for idx, det in localizable:
            center = np.asarray(det.center_xyz, dtype=np.float64)
            best_track: _Track | None = None
            best_dist = cfg.assoc_max_dist_m
            for track in self._tracks:
                if track.class_name != det.class_name or id(track) in used_tracks:
                    continue
                dist = float(np.linalg.norm(track.filtered_center() - center))
                if dist <= best_dist:
                    best_dist = dist
                    best_track = track
            if best_track is None:
                best_track = _Track(track_id=self._next_id, class_name=det.class_name)
                self._next_id += 1
                self._tracks.append(best_track)
            used_tracks.add(id(best_track))
            best_track.add(det, cfg.window_size)
            matched_track_for_idx[idx] = best_track

        # Update track misses and drop stale tracks.
        for track in self._tracks:
            if id(track) not in used_tracks:
                track.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= cfg.max_misses]

        # Emit one TrackedDetection per visible detection, in input order.
        return [self._build_tracked(det, matched_track_for_idx[idx]) for idx, det in localizable]

    def _build_tracked(self, det: Any, track: _Track) -> TrackedDetection:
        cfg = self.config
        center = track.filtered_center()
        pos_std = track.position_std_m()
        yaw_deg = track.filtered_yaw_deg()
        yaw_std = track.yaw_std_deg()
        conf_mean = track.confidence_mean()
        in_ws = bool(getattr(det, "in_workspace", True))
        yaw_reliable = track.yaw_reliable()

        # Symmetric objects have no meaningful yaw.
        yaw_ok_strict = (yaw_std <= cfg.max_yaw_std_deg) if yaw_reliable else True
        yaw_ok_loose = (yaw_std <= cfg.max_yaw_std_deg * cfg.stable_hysteresis) if yaw_reliable else True

        # Hysteresis + debounce so the flag does not flicker for borderline objects.
        enter_ok = (
            track.hits >= cfg.min_hits
            and pos_std <= cfg.max_position_std_m
            and yaw_ok_strict
            and conf_mean >= cfg.min_confidence
            and in_ws
        )
        if track.stable_latched:
            stay_ok = (
                in_ws
                and track.hits >= cfg.min_hits
                and pos_std <= cfg.max_position_std_m * cfg.stable_hysteresis
                and yaw_ok_loose
                and conf_mean >= cfg.min_confidence - cfg.stable_conf_margin
            )
            if not stay_ok:
                track.stable_latched = False
                track.enter_streak = 0
        else:
            if enter_ok:
                track.enter_streak += 1
                if track.enter_streak >= cfg.stable_enter_frames:
                    track.stable_latched = True
            else:
                track.enter_streak = 0
        stable = track.stable_latched

        # Freeze yaw to table-aligned (0) when not reliable, so the box/label do not
        # show a spinning, meaningless orientation.
        out_yaw_deg = yaw_deg if yaw_reliable else 0.0

        return TrackedDetection(
            raw=det,
            track_id=track.track_id,
            hits=track.hits,
            confidence_mean=conf_mean,
            center_filtered_xyz=[float(v) for v in center],
            extent_filtered_xyz=[float(v) for v in track.filtered_extent()],
            yaw_filtered_deg=out_yaw_deg,
            yaw_filtered_rad=float(np.radians(out_yaw_deg)),
            position_std_m=pos_std,
            yaw_std_deg=yaw_std,
            yaw_reliable=yaw_reliable,
            stable=stable,
        )
