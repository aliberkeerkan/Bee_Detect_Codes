import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from bee_detector import BeeDetector
from bee_follower import BeeFollower
from kalman_filter import KalmanFilter


class TrackingEngine:
    """
    ArUco-ID-based tracking engine.

    Pipeline
    --------
    red-channel background subtraction -> Otsu threshold -> dilation -> blob boxes
    -> BeeDetector on blob crops -> best detection per ID -> per-ID Kalman prediction
    """

    def __init__(self, settings: dict):
        self.settings = settings

        self.video_path = settings["video_path"]
        self.background_path = settings["background_path"]
        self.aruco_dict_path = settings["aruco_dict_path"]

        self.start_time_s = settings["start_time_s"]
        self.end_time_s = settings["end_time_s"]
        self.fps = float(settings.get("fps", 30.0) or 30.0)
        self.start_frame = int(self.start_time_s * self.fps)
        self.end_frame = int(self.end_time_s * self.fps)

        self.arenas = list(settings.get("arenas", []))
        self.stimulus_areas = list(settings.get("stimulus_areas", []))
        self.pixel_to_mm = float(settings.get("pixel_to_mm", 1.0))

        self.gui_min_area = float(settings.get("min_area", 0))
        self.gui_max_area = float(settings.get("max_area", 1_000_000))

        self.detector_settings = dict(settings.get("bee_detector_settings", {}))
        self.follower_settings = dict(settings.get("bee_follower_settings", {}))

        # Keep setup min/max area as the effective follower thresholds.
        self.follower_settings["min_area"] = float(self.gui_min_area)
        self.follower_settings["max_area"] = float(self.gui_max_area)

        # Kalman settings intentionally mirror detector_test.py defaults.
        self.max_missed_frames = int(settings.get("kalman_max_missed_frames", 10))
        self.kalman_process_noise = float(settings.get("kalman_process_noise", 1.0))
        self.kalman_measurement_noise = float(settings.get("kalman_measurement_noise", 0.0))

        # Morphology settings intentionally fixed unless later exposed in GUI.
        self.dilate_kernel_size = int(settings.get("dilate_kernel_size", 5))
        self.dilate_iterations = int(settings.get("dilate_iterations", 1))

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.marker_ids = self._read_marker_ids_from_dictionary(self.aruco_dict_path)
        self.id_to_index = {marker_id: i for i, marker_id in enumerate(self.marker_ids)}

        self.raw_angles: List[List[float]] = []
        self.filtered_angles: List[List[float]] = []
        self.filtered_is_prediction: List[List[int]] = []
        self.last_processed_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_tracking(self, progress_callback=None):
        background = self._imread_unicode(self.background_path, cv2.IMREAD_UNCHANGED)
        if background is None:
            raise RuntimeError(f"Could not load background image: {self.background_path}")

        bg_h, bg_w = background.shape[:2]
        if (bg_w, bg_h) != (self.frame_width, self.frame_height):
            raise RuntimeError(
                f"Background size {bg_w}x{bg_h} does not match video frame size "
                f"{self.frame_width}x{self.frame_height}."
            )

        bg_red = self._extract_red_like_channel(background)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        total_frames = max(0, self.end_frame - self.start_frame)
        if total_frames <= 0:
            cap.release()
            return [], []

        arena_mask = self._create_shape_mask(self.arenas)

        follower = BeeFollower(
            min_area=self.follower_settings.get("min_area", 50.0),
            max_area=self.follower_settings.get("max_area", 5000.0),
        )

        detector = BeeDetector(self.aruco_dict_path)
        detector.set_debug(show_debug=False, debug_marker_id=0)
        detector.set_candidate_settings(
            min_area=self.detector_settings.get("min_area"),
            max_contour_area=self.detector_settings.get("max_contour_area"),
            min_solidity=self.detector_settings.get("min_solidity"),
            poly_eps_ratio=self.detector_settings.get("poly_eps_ratio"),
            min_side_pixels=self.detector_settings.get("min_side_pixels"),
            max_side_ratio=self.detector_settings.get("max_side_ratio"),
            dedup_center_thresh=self.detector_settings.get("dedup_center_thresh"),
            remove_small_group_area=self.detector_settings.get("remove_small_group_area"),
            remove_large_group_area=self.detector_settings.get("remove_large_group_area"),
        )
        detector.set_decode_settings(
            max_hamming=self.detector_settings.get("max_hamming"),
            max_border_errors=self.detector_settings.get("max_border_errors"),
        )
        if "candidate_thresh" in self.detector_settings and self.detector_settings["candidate_thresh"] is not None:
            detector.candidate_thresh = int(self.detector_settings["candidate_thresh"])

        if self.dilate_kernel_size > 1:
            dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilate_kernel_size, self.dilate_kernel_size),
            )
        else:
            dilate_kernel = None

        tracks: Dict[int, Dict[str, object]] = {}
        raw_coords_history: List[List[Tuple[int, int]]] = []
        filtered_coords_history: List[List[Tuple[int, int]]] = []

        self.raw_angles = []
        self.filtered_angles = []
        self.filtered_is_prediction = []

        for frame_count in range(total_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            self.last_processed_frame = frame.copy()
            frame_red = self._extract_red_like_channel(frame)

            diff = cv2.absdiff(frame_red, bg_red)
            _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Only keep motion inside arenas.
            binary = cv2.bitwise_and(binary, arena_mask)

            if dilate_kernel is not None:
                binary_dilated = cv2.dilate(binary, dilate_kernel, iterations=self.dilate_iterations)
            else:
                binary_dilated = binary.copy()

            # Prevent dilation bleed outside arenas.
            binary_dilated = cv2.bitwise_and(binary_dilated, arena_mask)

            blobs = follower.detect(binary_dilated)
            detections_by_id = self._gather_best_detections(frame_red, blobs, detector)

            # Predict all existing tracks once for this new frame.
            for marker_id, track in list(tracks.items()):
                px, py = track["kf"].predict()
                track["pred_pos"] = (int(round(px)), int(round(py)))
                track["missed"] += 1
                track["updated_this_frame"] = False

            # Update tracks with real detections.
            for marker_id, det in detections_by_id.items():
                mx, my = det["center"]

                if marker_id not in tracks:
                    kf = KalmanFilter(
                        dt=1.0 / self.fps,
                        process_noise=self.kalman_process_noise,
                        measurement_noise=self.kalman_measurement_noise,
                    )
                    cx, cy = kf.update(mx, my)
                    tracks[marker_id] = {
                        "kf": kf,
                        "pred_pos": (int(round(cx)), int(round(cy))),
                        "missed": 0,
                        "updated_this_frame": True,
                        "last_angle_deg": float(det["angle_deg"]),
                    }
                else:
                    cx, cy = tracks[marker_id]["kf"].update(mx, my)
                    tracks[marker_id]["pred_pos"] = (int(round(cx)), int(round(cy)))
                    tracks[marker_id]["missed"] = 0
                    tracks[marker_id]["updated_this_frame"] = True
                    tracks[marker_id]["last_angle_deg"] = float(det["angle_deg"])

            stale_ids = [marker_id for marker_id, track in tracks.items() if track["missed"] > self.max_missed_frames]
            for marker_id in stale_ids:
                del tracks[marker_id]

            frame_raw_coords: List[Tuple[int, int]] = []
            frame_filtered_coords: List[Tuple[int, int]] = []
            frame_raw_angles: List[float] = []
            frame_filtered_angles: List[float] = []
            frame_pred_flags: List[int] = []

            for marker_id in self.marker_ids:
                det = detections_by_id.get(marker_id)
                track = tracks.get(marker_id)

                if det is not None:
                    cx, cy = det["center"]
                    frame_raw_coords.append((int(round(cx)), int(round(cy))))
                    frame_raw_angles.append(float(det["angle_deg"]))
                else:
                    frame_raw_coords.append((-1, -1))
                    frame_raw_angles.append(-1.0)

                if track is not None:
                    fx, fy = track["pred_pos"]
                    frame_filtered_coords.append((int(round(fx)), int(round(fy))))
                    frame_filtered_angles.append(float(track["last_angle_deg"]))
                    frame_pred_flags.append(0 if track["updated_this_frame"] else 1)
                else:
                    frame_filtered_coords.append((-1, -1))
                    frame_filtered_angles.append(-1.0)
                    frame_pred_flags.append(0)

            raw_coords_history.append(frame_raw_coords)
            filtered_coords_history.append(frame_filtered_coords)
            self.raw_angles.append(frame_raw_angles)
            self.filtered_angles.append(frame_filtered_angles)
            self.filtered_is_prediction.append(frame_pred_flags)

            if progress_callback is not None:
                progress_callback(int(100 * (frame_count + 1) / total_frames))

        cap.release()
        return raw_coords_history, filtered_coords_history

    def analyze_results(self, filtered_coords):
        # Placeholder for future stimulus-area analysis.
        return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_marker_ids_from_dictionary(self, dict_path: str) -> List[int]:
        detector = BeeDetector(dict_path)
        return sorted(int(marker_id) for marker_id in detector.custom_markers.keys())

    @staticmethod
    def _imread_unicode(path: str, flags: int):
        try:
            data = np.fromfile(path, dtype=np.uint8)
        except Exception:
            return cv2.imread(path, flags)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)

    @staticmethod
    def _extract_red_like_channel(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.copy()
        if image.ndim == 3 and image.shape[2] >= 3:
            return image[:, :, 2].copy()
        raise ValueError("Expected grayscale or BGR image for channel extraction.")

    def _create_shape_mask(self, shapes: List[dict]) -> np.ndarray:
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        if not shapes:
            return mask

        for shape_data in shapes:
            geom = shape_data["geom"]
            if shape_data["shape"] == "circle":
                cv2.circle(mask, (geom[0], geom[1]), geom[2], 255, -1)
            elif shape_data["shape"] == "rect":
                cv2.rectangle(mask, (geom[0], geom[1]), (geom[0] + geom[2], geom[1] + geom[3]), 255, -1)
            elif shape_data["shape"] == "poly":
                cv2.fillPoly(mask, [np.array(geom, dtype=np.int32)], 255)
        return mask

    @staticmethod
    def _crop_channel(channel_img: np.ndarray, bbox: Tuple[int, int, int, int]):
        x, y, w, h = bbox
        H, W = channel_img.shape[:2]

        x = max(0, int(x))
        y = max(0, int(y))
        x2 = min(W, x + int(w))
        y2 = min(H, y + int(h))

        if x2 <= x or y2 <= y:
            return None, None

        crop = channel_img[y:y2, x:x2].copy()
        return crop, (x, y)

    @staticmethod
    def _local_points_to_global(points_local: np.ndarray, crop_origin: Tuple[int, int]) -> np.ndarray:
        ox, oy = crop_origin
        pts = np.asarray(points_local, dtype=np.float32).copy()
        pts[:, 0] += ox
        pts[:, 1] += oy
        return pts

    @staticmethod
    def _detection_score(det: dict) -> Tuple[int, int]:
        return (
            int(det.get("hamming", 10**9)),
            int(det.get("border_errors", 10**9)),
        )

    def _gather_best_detections(self, red_channel: np.ndarray, blobs: List[dict], detector: BeeDetector) -> Dict[int, dict]:
        best_by_id: Dict[int, dict] = {}

        for blob in blobs:
            crop_red, crop_origin = self._crop_channel(red_channel, blob["bbox"])
            if crop_red is None:
                continue

            det_result = detector.detect(crop_red)
            detections = det_result["detections"]

            for det in detections:
                local_corners = np.asarray(det["marker_corners"], dtype=np.float32)
                global_corners = self._local_points_to_global(local_corners, crop_origin)
                center = global_corners.mean(axis=0)

                candidate = {
                    "id": int(det["id"]),
                    "angle_deg": float(det["angle_deg"]),
                    "global_corners": global_corners,
                    "center": (float(center[0]), float(center[1])),
                    "hamming": int(det["hamming"]),
                    "border_errors": int(det["border_errors"]),
                }

                marker_id = candidate["id"]
                if marker_id not in best_by_id or self._detection_score(candidate) < self._detection_score(best_by_id[marker_id]):
                    best_by_id[marker_id] = candidate

        return best_by_id

    def _is_point_in_shape(self, point, shape_data):
        x, y = point
        shape = shape_data["shape"]
        geom = shape_data["geom"]
        if shape == "circle":
            return (x - geom[0]) ** 2 + (y - geom[1]) ** 2 < geom[2] ** 2
        if shape == "rect":
            return geom[0] <= x <= geom[0] + geom[2] and geom[1] <= y <= geom[1] + geom[3]
        if shape == "poly":
            return cv2.pointPolygonTest(np.array(geom, dtype=np.int32), (int(x), int(y)), False) >= 0
        return False
