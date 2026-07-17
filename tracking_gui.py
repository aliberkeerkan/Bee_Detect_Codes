import sys
import os
import cv2
import math
import numpy as np
from datetime import date
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt

from tracking_core import TrackingEngine
from background_builder import BackgroundBuilder


DEFAULT_BEE_DETECTOR_SETTINGS = {
    "candidate_thresh": 90,
    "min_area": 100.0,
    "max_contour_area": 500.0,
    "min_solidity": 0.2,
    "poly_eps_ratio": 0.05,
    "min_side_pixels": 3.0,
    "max_side_ratio": 2.2,
    "dedup_center_thresh": 12.0,
    "remove_small_group_area": None,
    "remove_large_group_area": None,
    "max_hamming": 0,
    "max_border_errors": 10,
}

DEFAULT_BEE_FOLLOWER_SETTINGS = {
    "min_area": 50.0,
    "max_area": 5000.0,
}


def parse_time_str(time_str: str) -> int:
    try:
        h, m, s = map(int, time_str.split(":"))
        return h * 3600 + m * 60 + s
    except ValueError:
        return 0



def today_experiment_name() -> str:
    return date.today().isoformat()



def get_shape_center(shape_data):
    geom = shape_data["geom"]
    if shape_data["shape"] == "circle":
        return (geom[0], geom[1])
    if shape_data["shape"] == "rect":
        return (geom[0] + geom[2] // 2, geom[1] + geom[3] // 2)
    if shape_data["shape"] == "poly":
        pts = np.array(geom)
        return (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))
    return (0, 0)


class OptionalFloatEdit(QtWidgets.QLineEdit):
    def __init__(self, value=None, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("None")
        self.set_value(value)

    def value(self):
        text = self.text().strip()
        if text == "":
            return None
        return float(text)

    def set_value(self, value):
        self.setText("" if value is None else str(value))


class DetectionSettingsDialog(QtWidgets.QDialog):
    def __init__(self, detector_settings: dict, follower_settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Settings")
        self.setModal(True)

        self._detector_widgets = {}
        self._follower_widgets = {}

        main_layout = QtWidgets.QVBoxLayout(self)

        detector_candidate_group = QtWidgets.QGroupBox("BeeDetector — Candidate Extraction")
        detector_candidate_layout = QtWidgets.QFormLayout(detector_candidate_group)
        main_layout.addWidget(detector_candidate_group)

        self._add_spin(detector_candidate_layout, self._detector_widgets, "candidate_thresh", 0, 255, 0)
        self._add_double_spin(detector_candidate_layout, self._detector_widgets, "min_area", 0.0, 1_000_000.0, 2)
        self._add_double_spin(detector_candidate_layout, self._detector_widgets, "max_contour_area", 0.0, 1_000_000.0, 2)
        self._add_double_spin(detector_candidate_layout, self._detector_widgets, "min_solidity", 0.0, 1.0, 4, 0.01)
        self._add_double_spin(detector_candidate_layout, self._detector_widgets, "poly_eps_ratio", 0.0, 1.0, 4, 0.005)
        self._add_double_spin(detector_candidate_layout, self._detector_widgets, "min_side_pixels", 0.0, 10_000.0, 2)
        self._add_double_spin(detector_candidate_layout, self._detector_widgets, "max_side_ratio", 0.0, 100.0, 3, 0.05)
        self._add_double_spin(detector_candidate_layout, self._detector_widgets, "dedup_center_thresh", 0.0, 10_000.0, 2)
        self._add_optional_float(detector_candidate_layout, self._detector_widgets, "remove_small_group_area")
        self._add_optional_float(detector_candidate_layout, self._detector_widgets, "remove_large_group_area")

        detector_decode_group = QtWidgets.QGroupBox("BeeDetector — Decode")
        detector_decode_layout = QtWidgets.QFormLayout(detector_decode_group)
        main_layout.addWidget(detector_decode_group)

        self._add_spin(detector_decode_layout, self._detector_widgets, "max_hamming", 0, 100, 0)
        self._add_spin(detector_decode_layout, self._detector_widgets, "max_border_errors", 0, 1000, 0)

        follower_group = QtWidgets.QGroupBox("BeeFollower")
        follower_layout = QtWidgets.QFormLayout(follower_group)
        main_layout.addWidget(follower_group)

        self._add_double_spin(follower_layout, self._follower_widgets, "min_area", 0.0, 1_000_000.0, 2)
        self._add_double_spin(follower_layout, self._follower_widgets, "max_area", 0.0, 1_000_000.0, 2)

        button_row = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("Reset to Defaults")
        button_row.addWidget(self.btn_reset)
        button_row.addStretch()
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_row.addWidget(self.button_box)
        main_layout.addLayout(button_row)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.btn_reset.clicked.connect(self.reset_to_defaults)

        self.set_values(detector_settings, follower_settings)
        self.resize(480, 520)

    def _add_spin(self, layout, store, key, minimum, maximum, decimals):
        widget = QtWidgets.QSpinBox()
        widget.setRange(int(minimum), int(maximum))
        store[key] = widget
        layout.addRow(key, widget)

    def _add_double_spin(self, layout, store, key, minimum, maximum, decimals, step=0.1):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setKeyboardTracking(False)
        store[key] = widget
        layout.addRow(key, widget)

    def _add_optional_float(self, layout, store, key):
        widget = OptionalFloatEdit()
        store[key] = widget
        layout.addRow(key, widget)

    def set_values(self, detector_settings: dict, follower_settings: dict):
        for key, widget in self._detector_widgets.items():
            value = detector_settings.get(key, DEFAULT_BEE_DETECTOR_SETTINGS.get(key))
            if isinstance(widget, OptionalFloatEdit):
                widget.set_value(value)
            else:
                widget.setValue(value)

        for key, widget in self._follower_widgets.items():
            value = follower_settings.get(key, DEFAULT_BEE_FOLLOWER_SETTINGS.get(key))
            widget.setValue(value)

    def reset_to_defaults(self):
        self.set_values(DEFAULT_BEE_DETECTOR_SETTINGS, DEFAULT_BEE_FOLLOWER_SETTINGS)

    def values(self):
        detector = {}
        follower = {}

        for key, widget in self._detector_widgets.items():
            detector[key] = widget.value() if isinstance(widget, OptionalFloatEdit) else widget.value()

        for key, widget in self._follower_widgets.items():
            follower[key] = widget.value()

        return detector, follower


class InteractiveViewer(QtWidgets.QGraphicsView):
    shapes_updated = QtCore.Signal()
    line_drawn = QtCore.Signal(float, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)

        self.draw_config = {}
        self.current_shape_item = None
        self.start_pos = None
        self.drawing_poly_points = []
        self.item_map = {}

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Delete:
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    def set_frame(self, frame_bgr):
        if frame_bgr is None:
            return
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(frame_bgr.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self._pixmap_item.setPixmap(pixmap)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def start_drawing(self, shape, category):
        self.draw_config = {"shape": shape, "category": category}
        if shape == "poly":
            self.drawing_poly_points = []
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def finish_drawing(self):
        shape = self.draw_config.get("shape")
        if self.current_shape_item and shape == "poly" and self.drawing_poly_points:
            self.scene().removeItem(self.current_shape_item)
            self.current_shape_item = None

        if shape == "poly" and len(self.drawing_poly_points) > 2:
            geom = [(p.x(), p.y()) for p in self.drawing_poly_points]
            data = {"shape": "poly", "category": self.draw_config.get("category"), "geom": geom}
            self.add_shape_item(data)

        self.drawing_poly_points = []
        self.draw_config = {}
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.shapes_updated.emit()

    def add_shape_item(self, data):
        category = data.get("category")
        shape = data.get("shape")
        geom = data.get("geom")

        color = QtGui.QColor(0, 150, 255) if category == "arena" else QtGui.QColor(255, 200, 0)
        pen = QtGui.QPen(color, 2, QtCore.Qt.PenStyle.SolidLine)
        brush = QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 50))

        item = None
        if shape == "circle":
            cx, cy, r = geom
            rect = QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r)
            item = self.scene().addEllipse(rect, pen, brush)
        elif shape == "rect":
            x, y, w, h = geom
            item = self.scene().addRect(QtCore.QRectF(x, y, w, h), pen, brush)
        elif shape == "poly":
            polygon = QtGui.QPolygonF([QtCore.QPointF(p[0], p[1]) for p in geom])
            item = self.scene().addPolygon(polygon, pen, brush)

        if item is not None:
            item.setFlags(
                QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            )
            item.setData(0, data)
            self.item_map[id(data)] = item
            text_item = QtWidgets.QGraphicsSimpleTextItem(item)
            text_item.setBrush(QtCore.Qt.GlobalColor.white)
            text_item.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Weight.Bold))
            text_item.setZValue(1)
        return item

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        if not self.draw_config:
            super().mousePressEvent(event)
            return

        shape = self.draw_config.get("shape")
        if shape in ["circle", "rect", "line", "measure_length", "measure_width"] or shape.startswith("measure_diameter_"):
            self.start_pos = pos
        elif shape == "poly":
            self.drawing_poly_points.append(pos)
            if len(self.drawing_poly_points) > 1:
                if self.current_shape_item:
                    self.scene().removeItem(self.current_shape_item)
                pen = QtGui.QPen(QtGui.QColor(255, 255, 0, 200), 2, QtCore.Qt.PenStyle.DashLine)
                poly = QtGui.QPolygonF(self.drawing_poly_points)
                self.current_shape_item = self.scene().addPolygon(poly, pen)

    def mouseMoveEvent(self, event):
        if not self.start_pos or self.draw_config.get("shape") == "poly":
            super().mouseMoveEvent(event)
            return

        if self.current_shape_item:
            self.scene().removeItem(self.current_shape_item)
        pos = self.mapToScene(event.position().toPoint())
        pen = QtGui.QPen(QtGui.QColor(255, 255, 0, 200), 2, QtCore.Qt.PenStyle.DashLine)
        shape = self.draw_config.get("shape")

        if shape == "rect":
            self.current_shape_item = self.scene().addRect(QtCore.QRectF(self.start_pos, pos).normalized(), pen)
        elif shape == "circle":
            radius = math.sqrt((pos.x() - self.start_pos.x()) ** 2 + (pos.y() - self.start_pos.y()) ** 2)
            rect = QtCore.QRectF(self.start_pos.x() - radius, self.start_pos.y() - radius, 2 * radius, 2 * radius)
            self.current_shape_item = self.scene().addEllipse(rect, pen)
        elif shape in ["line", "measure_length", "measure_width"] or shape.startswith("measure_diameter_"):
            self.current_shape_item = self.scene().addLine(QtCore.QLineF(self.start_pos, pos), pen)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

        if self.draw_config and self.start_pos:
            shape = self.draw_config.get("shape")
            category = self.draw_config.get("category")

            if shape in ["circle", "rect"] and self.current_shape_item:
                scene_br = self.current_shape_item.sceneBoundingRect()
                if shape == "circle":
                    center = scene_br.center()
                    radius = scene_br.width() / 2.0
                    geom = [int(center.x()), int(center.y()), int(radius)]
                else:
                    geom = [int(scene_br.x()), int(scene_br.y()), int(scene_br.width()), int(scene_br.height())]
                data = {"shape": shape, "category": category, "geom": geom}
                self.add_shape_item(data)
                self.shapes_updated.emit()

            elif shape in ["line", "measure_length", "measure_width"] or shape.startswith("measure_diameter_"):
                end_pos = self.mapToScene(event.position().toPoint())
                length = math.sqrt((end_pos.x() - self.start_pos.x()) ** 2 + (end_pos.y() - self.start_pos.y()) ** 2)
                self.line_drawn.emit(length, shape)

            if self.current_shape_item:
                self.scene().removeItem(self.current_shape_item)
            if shape != "poly":
                self.current_shape_item = None
                self.start_pos = None
                self.draw_config = {}
                self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        else:
            self.shapes_updated.emit()

    def wheelEvent(self, event):
        selected = self.scene().selectedItems()
        if not selected:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            return

        item = selected[0]
        data = item.data(0)
        if data and data["shape"] == "circle":
            delta = 5 if event.angleDelta().y() > 0 else -5
            scene_br = item.sceneBoundingRect()
            center = scene_br.center()
            new_r = max(5, scene_br.width() / 2 + delta)
            item.setPos(center.x() - new_r, center.y() - new_r)
            item.setRect(0, 0, 2 * new_r, 2 * new_r)
            data["geom"] = [int(center.x()), int(center.y()), int(new_r)]
            item.setData(0, data)
            self.shapes_updated.emit()

    def get_all_shapes_data(self):
        def _center_and_vsize(data):
            geom = data["geom"]
            if data["shape"] == "circle":
                return (int(geom[0]), int(geom[1])), float(geom[2])
            if data["shape"] == "rect":
                x, y, w, h = geom
                return (int(x + w // 2), int(y + h // 2)), float(h / 2.0)
            if data["shape"] == "poly":
                xs = [p[0] for p in geom]
                ys = [p[1] for p in geom]
                cx = int(sum(xs) / len(xs))
                cy = int(sum(ys) / len(ys))
                vsize = (max(ys) - min(ys)) / 2.0 if len(ys) >= 2 else 20.0
                return (cx, cy), float(vsize)
            return (0, 0), 20.0

        arena_items, stim_items = [], []
        for item in self.scene().items():
            data = item.data(0)
            if not data or "category" not in data:
                continue

            scene_br = item.sceneBoundingRect()
            if data["shape"] == "circle":
                center = scene_br.center()
                data["geom"] = [int(center.x()), int(center.y()), int(scene_br.width() / 2)]
            elif data["shape"] == "rect":
                data["geom"] = [int(scene_br.x()), int(scene_br.y()), int(scene_br.width()), int(scene_br.height())]
            elif data["shape"] == "poly":
                scene_poly = item.mapToScene(item.polygon())
                data["geom"] = [(int(p.x()), int(p.y())) for p in scene_poly]

            center, vsize = _center_and_vsize(data)
            rec = (item, data, center, vsize)
            if data["category"] == "arena":
                arena_items.append(rec)
            elif data["category"] == "stim":
                stim_items.append(rec)

        def _row_major_sort(recs):
            if not recs:
                return []
            vs = sorted(v for *_, v in recs)
            tol = vs[len(vs) // 2] if vs else 20.0
            tol = max(10.0, tol * 0.75)
            recs_sorted = sorted(recs, key=lambda r: (r[2][1], r[2][0]))
            rows = []
            for rec in recs_sorted:
                placed = False
                for row in rows:
                    mean_y = sum(rr[2][1] for rr in row) / len(row)
                    if abs(rec[2][1] - mean_y) <= tol:
                        row.append(rec)
                        placed = True
                        break
                if not placed:
                    rows.append([rec])
            rows.sort(key=lambda row: sum(rr[2][1] for rr in row) / len(row))
            flat = []
            for row in rows:
                row.sort(key=lambda rr: rr[2][0])
                flat.extend(row)
            return flat

        arena_ordered = _row_major_sort(arena_items)
        stim_ordered = _row_major_sort(stim_items)

        def _apply_ids(ordered, prefix):
            out = []
            for i, (item, data, _, _) in enumerate(ordered, start=1):
                data["id"] = f"{prefix}{i}"
                text_item = next((c for c in item.childItems() if isinstance(c, QtWidgets.QGraphicsSimpleTextItem)), None)
                if text_item is None:
                    text_item = QtWidgets.QGraphicsSimpleTextItem(item)
                    text_item.setBrush(QtCore.Qt.GlobalColor.white)
                    text_item.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Weight.Bold))
                    text_item.setZValue(1)
                text_item.setText(data["id"])
                sr = item.boundingRect()
                tr = text_item.boundingRect()
                text_item.setPos(sr.center().x() - tr.width() / 2, sr.center().y() - tr.height() / 2)
                out.append(data)
            return out

        arenas_data = _apply_ids(arena_ordered, "A")
        stims_data = _apply_ids(stim_ordered, "S")
        return arenas_data + stims_data

    def delete_selected(self):
        items_to_remove = self.scene().selectedItems()
        if not items_to_remove:
            return
        for item in items_to_remove:
            data = item.data(0)
            if data and id(data) in self.item_map:
                del self.item_map[id(data)]
            self.scene().removeItem(item)
        self.shapes_updated.emit()

    def clear_shapes_by_category(self, category_to_clear):
        items_to_remove = []
        data_ids_to_remove = set()
        for item in self.scene().items():
            data = item.data(0)
            if data and data.get("category") == category_to_clear:
                items_to_remove.append(item)
                data_ids_to_remove.add(id(data))
        if not items_to_remove:
            return
        for item in items_to_remove:
            self.scene().removeItem(item)
        for data_id in data_ids_to_remove:
            if data_id in self.item_map:
                del self.item_map[data_id]
        self.shapes_updated.emit()


class CanvasWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Canvas")
        self.viewer = InteractiveViewer(self)
        self.setCentralWidget(self.viewer)
        self.resize(800, 600)


class AnalysisWorker(QtCore.QThread):
    progress = QtCore.Signal(int)
    # Named result_ready (not "finished") to avoid shadowing QThread's own
    # built-in `finished` signal, which Qt relies on for thread lifecycle
    # bookkeeping - reusing the name causes "QThread: Destroyed while thread
    # is still running" aborts on shutdown (same bug found/fixed in analysis_gui.py).
    result_ready = QtCore.Signal(object, object)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def run(self):
        try:
            engine = TrackingEngine(self.settings)
            raw, filtered = engine.run_tracking(self.progress.emit)
            analysis_results = engine.analyze_results(filtered)
            self.result_ready.emit(
                {
                    "raw_coords": raw,
                    "filtered_coords": filtered,
                    "raw_angles": engine.raw_angles,
                    "filtered_angles": engine.filtered_angles,
                    "filtered_is_prediction": engine.filtered_is_prediction,
                    "marker_ids": engine.marker_ids,
                    "analysis": analysis_results,
                    "engine": engine,
                },
                None,
            )
        except Exception as e:
            self.result_ready.emit(None, e)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.settings = {}
        self.detector_settings = dict(DEFAULT_BEE_DETECTOR_SETTINGS)
        self.follower_settings = dict(DEFAULT_BEE_FOLLOWER_SETTINGS)
        self.measured_length = None
        self.measured_width = None
        self.worker = None
        self.canvas = CanvasWindow()
        self.controls = self.create_control_window()
        self.canvas.show()
        self.controls.show()
        self.update_background_buttons()

    def _create_autodetect_widget(self, category):
        group = QtWidgets.QGroupBox("Autodetection")
        layout = QtWidgets.QFormLayout(group)
        diameter_widget = QtWidgets.QWidget()
        diameter_layout = QtWidgets.QHBoxLayout(diameter_widget)
        diameter_layout.setContentsMargins(0, 0, 0, 0)
        edit_diameter = QtWidgets.QLineEdit("100")
        btn_measure_diameter = QtWidgets.QPushButton("Measure")
        diameter_layout.addWidget(edit_diameter)
        diameter_layout.addWidget(btn_measure_diameter)
        spin_threshold = QtWidgets.QSpinBox(minimum=0, maximum=255, value=127)
        btn_detect_circ = QtWidgets.QPushButton("Autodetect Circles")
        btn_detect_rect = QtWidgets.QPushButton("Autodetect Rectangles")
        btn_remove_category_areas = QtWidgets.QPushButton(f"Remove All {category.capitalize()} Areas")

        layout.addRow("Expected Diameter/Diagonal (px):", diameter_widget)
        layout.addRow("Detection Threshold (0-255):", spin_threshold)
        layout.addRow(btn_detect_circ)
        layout.addRow(btn_detect_rect)
        layout.addRow(btn_remove_category_areas)

        if category == "arena":
            self.edit_arena_diameter = edit_diameter
            self.spin_arena_threshold = spin_threshold
        else:
            self.edit_stim_diameter = edit_diameter
            self.spin_stim_threshold = spin_threshold

        btn_measure_diameter.clicked.connect(lambda: self.measure_for_autodetect(category))
        btn_detect_circ.clicked.connect(lambda: self.autodetect_shapes(category, "circle"))
        btn_detect_rect.clicked.connect(lambda: self.autodetect_shapes(category, "rect"))
        btn_remove_category_areas.clicked.connect(lambda: self.canvas.viewer.clear_shapes_by_category(category))
        return group

    def show_scrollable_help(self):
        help_text = """
        <h2>Bee Tracking GUI Help</h2>
        <h3>1. Setup</h3>
        <ul>
        <li><b>Load Video:</b> Select the experiment video.</li>
        <li><b>Open Dictionary:</b> Select the custom ArUco dictionary TXT file.</li>
        <li><b>Build BG:</b> Build a background image from the selected video and save it as <code>background.png</code> in the working directory.</li>
        <li><b>Open BG:</b> Open the saved background image in a separate window if it exists.</li>
        <li><b>Settings:</b> Open the detector/follower settings dialog. These values are passed to the core during tracking.</li>
        </ul>
        <h3>2. Define Areas</h3>
        <ul>
        <li>Manual drawing and autodetection work the same as before.</li>
        <li>Delete selected shapes with the Delete key.</li>
        </ul>
        <h3>3. Visualization</h3>
        <ul>
        <li>Save optional track video and trajectory image overlays.</li>
        </ul>
        <h3>4. Run</h3>
        <ul>
        <li>Run tracking after setting scale, defining at least one arena, selecting dictionary, and preparing a background.</li>
        </ul>
        """

        dialog = QtWidgets.QDialog(self.controls)
        dialog.setWindowTitle("Help")
        layout = QtWidgets.QVBoxLayout(dialog)
        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(help_text)
        layout.addWidget(text_edit)
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        dialog.resize(650, 500)
        dialog.exec()

    def create_control_window(self):
        win = QtWidgets.QMainWindow()
        win.setWindowTitle("Controls")

        self.btn_load_video = QtWidgets.QPushButton("Load Video")
        self.lbl_video_name = QtWidgets.QLabel("No video loaded.")
        self.lbl_video_name.setStyleSheet("font-style: italic;")

        self.edit_dict_path = QtWidgets.QLineEdit()
        self.edit_dict_path.setReadOnly(True)
        self.btn_open_dict = QtWidgets.QPushButton("Open Dictionary")

        self.edit_exp_name = QtWidgets.QLineEdit(today_experiment_name())
        self.edit_start_time = QtWidgets.QLineEdit("00:00:00")
        self.edit_end_time = QtWidgets.QLineEdit("00:00:10")
        self.edit_output_dir = QtWidgets.QLineEdit()
        self.btn_browse_dir = QtWidgets.QPushButton("Browse...")

        self.edit_scale_len = QtWidgets.QLineEdit("10.0")
        self.btn_draw_scale = QtWidgets.QPushButton("Draw Scale Line")
        self.lbl_scale = QtWidgets.QLabel("Scale: Not Set")

        self.btn_measure_len = QtWidgets.QPushButton("Measure Length")
        self.btn_measure_wid = QtWidgets.QPushButton("Measure Width")
        self.lbl_animal_size = QtWidgets.QLabel("Animal Size: Not Set")

        self.spin_min_area = QtWidgets.QSpinBox(minimum=0, maximum=1_000_000, value=100)
        self.spin_max_area = QtWidgets.QSpinBox(minimum=1, maximum=1_000_000, value=10000)

        self.spin_bg_num_frames = QtWidgets.QSpinBox(minimum=1, maximum=100000, value=50)
        self.edit_bg_start_frame = QtWidgets.QLineEdit()
        self.edit_bg_end_frame = QtWidgets.QLineEdit()
        self.chk_bg_grayscale = QtWidgets.QCheckBox("Build in grayscale")
        self.btn_build_bg = QtWidgets.QPushButton("Build BG")
        self.btn_open_bg = QtWidgets.QPushButton("Open BG")
        self.btn_settings = QtWidgets.QPushButton("Settings")

        self.btn_add_arena_circ = QtWidgets.QPushButton("Circle")
        self.btn_add_arena_rect = QtWidgets.QPushButton("Rect")
        self.btn_add_arena_poly = QtWidgets.QPushButton("Poly")
        self.btn_arena_poly_done = QtWidgets.QPushButton("Finish Poly")
        self.btn_add_stim_circ = QtWidgets.QPushButton("Circle")
        self.btn_add_stim_rect = QtWidgets.QPushButton("Rect")
        self.btn_add_stim_poly = QtWidgets.QPushButton("Poly")
        self.btn_stim_poly_done = QtWidgets.QPushButton("Finish Poly")

        self.btn_run = QtWidgets.QPushButton("Run Tracking & Analysis")
        self.progress_bar = QtWidgets.QProgressBar()
        self.chk_save_video = QtWidgets.QCheckBox("Save Track Video")
        self.chk_save_video.setChecked(True)
        self.chk_draw_raw_img = QtWidgets.QCheckBox("Draw Raw Trajectories (Image)")
        self.chk_draw_raw_img.setChecked(True)
        self.chk_draw_filtered_img = QtWidgets.QCheckBox("Draw Filtered Trajectories (Image)")
        self.chk_draw_filtered_img.setChecked(True)

        self.btn_help = QtWidgets.QPushButton("Help")
        self.btn_exit = QtWidgets.QPushButton("Exit Application")

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        win.setCentralWidget(scroll_area)
        main_widget = QtWidgets.QWidget()
        scroll_area.setWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)

        setup_group = QtWidgets.QGroupBox("1. Setup")
        setup_layout = QtWidgets.QFormLayout(setup_group)
        setup_layout.addRow(self.btn_load_video)
        setup_layout.addRow("Loaded Video:", self.lbl_video_name)

        dict_row = QtWidgets.QHBoxLayout()
        dict_row.addWidget(self.edit_dict_path)
        dict_row.addWidget(self.btn_open_dict)
        setup_layout.addRow("Dictionary:", dict_row)

        setup_layout.addRow("Experiment Name:", self.edit_exp_name)

        dir_row = QtWidgets.QHBoxLayout()
        dir_row.addWidget(self.edit_output_dir)
        dir_row.addWidget(self.btn_browse_dir)
        setup_layout.addRow("Output Directory:", dir_row)

        setup_layout.addRow("Start Time (h:m:s):", self.edit_start_time)
        setup_layout.addRow("End Time (h:m:s):", self.edit_end_time)

        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(QtWidgets.QLabel("Scale Length (mm):"))
        scale_row.addWidget(self.edit_scale_len)
        scale_row.addWidget(self.btn_draw_scale)
        setup_layout.addRow(scale_row)
        setup_layout.addRow(self.lbl_scale)

        measure_row = QtWidgets.QHBoxLayout()
        measure_row.addWidget(self.btn_measure_len)
        measure_row.addWidget(self.btn_measure_wid)
        setup_layout.addRow("Measure Animal Size:", measure_row)
        setup_layout.addRow(self.lbl_animal_size)

        area_row = QtWidgets.QHBoxLayout()
        area_row.addWidget(QtWidgets.QLabel("Min Area (px²):"))
        area_row.addWidget(self.spin_min_area)
        area_row.addWidget(QtWidgets.QLabel("Max Area (px²):"))
        area_row.addWidget(self.spin_max_area)
        setup_layout.addRow("Animal Size Threshold:", area_row)

        bg_group = QtWidgets.QGroupBox("Background Builder")
        bg_layout = QtWidgets.QFormLayout(bg_group)
        bg_layout.addRow("Frames to sample:", self.spin_bg_num_frames)
        bg_layout.addRow("Start frame (optional):", self.edit_bg_start_frame)
        bg_layout.addRow("End frame (optional):", self.edit_bg_end_frame)
        bg_layout.addRow(self.chk_bg_grayscale)
        bg_btn_row = QtWidgets.QHBoxLayout()
        bg_btn_row.addWidget(self.btn_build_bg)
        bg_btn_row.addWidget(self.btn_open_bg)
        bg_layout.addRow(bg_btn_row)
        setup_layout.addRow(bg_group)
        setup_layout.addRow(self.btn_settings)

        main_layout.addWidget(setup_group)

        tabs_group = QtWidgets.QGroupBox("2. Define Areas")
        tabs_layout = QtWidgets.QVBoxLayout(tabs_group)
        tab_widget = QtWidgets.QTabWidget()
        arena_tab = QtWidgets.QWidget()
        stim_tab = QtWidgets.QWidget()
        tab_widget.addTab(arena_tab, "Arenas")
        tab_widget.addTab(stim_tab, "Stimulus Areas")
        tabs_layout.addWidget(tab_widget)
        main_layout.addWidget(tabs_group)

        arena_layout = QtWidgets.QVBoxLayout(arena_tab)
        arena_manual_group = QtWidgets.QGroupBox("Manual Drawing")
        arena_manual_layout = QtWidgets.QHBoxLayout(arena_manual_group)
        arena_manual_layout.addWidget(self.btn_add_arena_circ)
        arena_manual_layout.addWidget(self.btn_add_arena_rect)
        arena_manual_layout.addWidget(self.btn_add_arena_poly)
        arena_manual_layout.addWidget(self.btn_arena_poly_done)
        arena_layout.addWidget(arena_manual_group)
        self.arena_autodetect_group = self._create_autodetect_widget("arena")
        arena_layout.addWidget(self.arena_autodetect_group)

        stim_layout = QtWidgets.QVBoxLayout(stim_tab)
        stim_manual_group = QtWidgets.QGroupBox("Manual Drawing")
        stim_manual_layout = QtWidgets.QHBoxLayout(stim_manual_group)
        stim_manual_layout.addWidget(self.btn_add_stim_circ)
        stim_manual_layout.addWidget(self.btn_add_stim_rect)
        stim_manual_layout.addWidget(self.btn_add_stim_poly)
        stim_manual_layout.addWidget(self.btn_stim_poly_done)
        stim_layout.addWidget(stim_manual_group)
        self.stim_autodetect_group = self._create_autodetect_widget("stim")
        stim_layout.addWidget(self.stim_autodetect_group)

        vis_group = QtWidgets.QGroupBox("3. Visualization")
        vis_layout = QtWidgets.QFormLayout(vis_group)
        vis_layout.addRow(self.chk_save_video)
        vis_layout.addRow(self.chk_draw_raw_img)
        vis_layout.addRow(self.chk_draw_filtered_img)
        main_layout.addWidget(vis_group)

        run_group = QtWidgets.QGroupBox("4. Run")
        run_layout = QtWidgets.QFormLayout(run_group)
        run_layout.addRow(self.btn_run)
        run_layout.addRow(self.progress_bar)
        main_layout.addWidget(run_group)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.btn_help)
        button_row.addStretch()
        button_row.addWidget(self.btn_exit)
        main_layout.addLayout(button_row)
        main_layout.addStretch()

        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_open_dict.clicked.connect(self.open_dictionary)
        self.btn_browse_dir.clicked.connect(self.select_output_directory)
        self.edit_start_time.editingFinished.connect(self.update_canvas_frame)
        self.btn_draw_scale.clicked.connect(self.draw_scale)
        self.btn_measure_len.clicked.connect(lambda: self.canvas.viewer.start_drawing("measure_length", "measure"))
        self.btn_measure_wid.clicked.connect(lambda: self.canvas.viewer.start_drawing("measure_width", "measure"))
        self.canvas.viewer.line_drawn.connect(self.handle_line_drawn)
        self.canvas.viewer.shapes_updated.connect(self.update_shape_lists)

        self.btn_build_bg.clicked.connect(self.build_background)
        self.btn_open_bg.clicked.connect(self.open_background)
        self.btn_settings.clicked.connect(self.open_settings_dialog)

        self.btn_add_arena_circ.clicked.connect(lambda: self.canvas.viewer.start_drawing("circle", "arena"))
        self.btn_add_arena_rect.clicked.connect(lambda: self.canvas.viewer.start_drawing("rect", "arena"))
        self.btn_add_arena_poly.clicked.connect(lambda: self.canvas.viewer.start_drawing("poly", "arena"))
        self.btn_arena_poly_done.clicked.connect(self.canvas.viewer.finish_drawing)
        self.btn_add_stim_circ.clicked.connect(lambda: self.canvas.viewer.start_drawing("circle", "stim"))
        self.btn_add_stim_rect.clicked.connect(lambda: self.canvas.viewer.start_drawing("rect", "stim"))
        self.btn_add_stim_poly.clicked.connect(lambda: self.canvas.viewer.start_drawing("poly", "stim"))
        self.btn_stim_poly_done.clicked.connect(self.canvas.viewer.finish_drawing)
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_help.clicked.connect(self.show_scrollable_help)
        self.btn_exit.clicked.connect(self.close_app)

        self.set_controls_enabled(False)
        self.btn_load_video.setEnabled(True)
        self.btn_open_dict.setEnabled(True)
        self.btn_help.setEnabled(True)
        self.btn_exit.setEnabled(True)
        self.edit_exp_name.setEnabled(True)
        return win

    def close_app(self):
        QtWidgets.QApplication.instance().quit()

    def set_controls_enabled(self, enabled):
        widgets = [
            self.edit_start_time,
            self.edit_end_time,
            self.edit_scale_len,
            self.btn_draw_scale,
            self.edit_output_dir,
            self.btn_browse_dir,
            self.btn_measure_len,
            self.btn_measure_wid,
            self.spin_min_area,
            self.spin_max_area,
            self.spin_bg_num_frames,
            self.edit_bg_start_frame,
            self.edit_bg_end_frame,
            self.chk_bg_grayscale,
            self.btn_build_bg,
            self.btn_open_bg,
            self.btn_settings,
            self.btn_add_arena_circ,
            self.btn_add_arena_rect,
            self.btn_add_arena_poly,
            self.btn_arena_poly_done,
            self.btn_add_stim_circ,
            self.btn_add_stim_rect,
            self.btn_add_stim_poly,
            self.btn_stim_poly_done,
            self.btn_run,
            self.chk_save_video,
            self.chk_draw_raw_img,
            self.chk_draw_filtered_img,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)
        self.arena_autodetect_group.setEnabled(enabled)
        self.stim_autodetect_group.setEnabled(enabled)

    def work_directory(self) -> Path | None:
        output_dir = self.edit_output_dir.text().strip()
        if output_dir:
            return Path(output_dir)
        video_path = self.settings.get("video_path")
        if video_path:
            return Path(video_path).parent
        return None

    def background_path(self) -> Path | None:
        work_dir = self.work_directory()
        if work_dir is None:
            return None
        return work_dir / "background.png"

    def update_background_buttons(self):
        bg_path = self.background_path()
        self.btn_open_bg.setEnabled(bg_path is not None and bg_path.exists())

    def show_image_window(self, window_name: str, image):
        if image is None:
            return
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

    def load_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.controls,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if not file_path:
            return
        self.settings["video_path"] = file_path
        self.lbl_video_name.setText(os.path.basename(file_path))
        if not self.edit_output_dir.text().strip():
            self.edit_output_dir.setText(os.path.dirname(file_path))
        self.update_canvas_frame()
        self.set_controls_enabled(True)
        self.btn_open_dict.setEnabled(True)
        self.edit_exp_name.setEnabled(True)
        self.update_background_buttons()

    def open_dictionary(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.controls,
            "Open ArUco Dictionary",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if not file_path:
            return
        self.edit_dict_path.setText(file_path)
        self.settings["aruco_dict_path"] = file_path

    def select_output_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self.controls, "Select Output Directory")
        if directory:
            self.edit_output_dir.setText(directory)
            self.update_background_buttons()

    def update_canvas_frame(self):
        if "video_path" not in self.settings:
            return
        cap = cv2.VideoCapture(self.settings["video_path"])
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self.controls, "Warning", "Could not open the selected video.")
            return
        self.settings["fps"] = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = int(parse_time_str(self.edit_start_time.text()) * self.settings["fps"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.settings["first_frame"] = frame
            self.canvas.viewer.set_frame(frame)
        else:
            QtWidgets.QMessageBox.warning(
                self.controls,
                "Warning",
                f"Could not seek to {self.edit_start_time.text()}. Showing first frame instead.",
            )
            self.edit_start_time.setText("00:00:00")
            self.update_canvas_frame()
        self.update_background_buttons()

    def build_background(self):
        video_path = self.settings.get("video_path")
        if not video_path:
            QtWidgets.QMessageBox.warning(self.controls, "Warning", "Load a video first.")
            return

        work_dir = self.work_directory()
        if work_dir is None:
            QtWidgets.QMessageBox.warning(self.controls, "Warning", "Select an output directory first.")
            return
        work_dir.mkdir(parents=True, exist_ok=True)
        bg_path = work_dir / "background.png"

        try:
            start_frame = int(self.edit_bg_start_frame.text()) if self.edit_bg_start_frame.text().strip() else 0
            end_frame = int(self.edit_bg_end_frame.text()) if self.edit_bg_end_frame.text().strip() else None
        except ValueError:
            QtWidgets.QMessageBox.warning(self.controls, "Input Error", "Start/End frame must be integers.")
            return

        try:
            builder = BackgroundBuilder(video_path)
            background = builder.build_and_save(
                output_path=str(bg_path),
                num_frames=self.spin_bg_num_frames.value(),
                start_frame=start_frame,
                end_frame=end_frame,
                grayscale=self.chk_bg_grayscale.isChecked(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self.controls, "Background Error", str(exc))
            return

        self.settings["background_path"] = str(bg_path)
        self.show_image_window("Background", background)
        self.update_background_buttons()
        QtWidgets.QMessageBox.information(self.controls, "Background Built", f"Saved background to:\n{bg_path}")

    def open_background(self):
        bg_path = self.background_path()
        if bg_path is None or not bg_path.exists():
            QtWidgets.QMessageBox.warning(self.controls, "Warning", "No background.png found in the working directory.")
            return
        image = cv2.imread(str(bg_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            QtWidgets.QMessageBox.warning(self.controls, "Warning", "Could not open background image.")
            return
        self.settings["background_path"] = str(bg_path)
        self.show_image_window("Background", image)
        self.update_background_buttons()

    def open_settings_dialog(self):
        dialog = DetectionSettingsDialog(self.detector_settings, self.follower_settings, self.controls)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            detector, follower = dialog.values()
            self.detector_settings = detector
            self.follower_settings = follower
            self.spin_min_area.setValue(int(round(float(follower.get("min_area", self.spin_min_area.value())))))
            self.spin_max_area.setValue(int(round(float(follower.get("max_area", self.spin_max_area.value())))))

    def draw_scale(self):
        self.canvas.viewer.start_drawing("line", "scale")

    def handle_line_drawn(self, pixel_length, line_type):
        if line_type == "line":
            try:
                real_length = float(self.edit_scale_len.text())
                if real_length > 0 and pixel_length > 0:
                    mm_per_pixel = real_length / pixel_length
                    self.settings["pixel_to_mm"] = mm_per_pixel
                    self.lbl_scale.setText(f"Scale: 1 mm = {1.0 / mm_per_pixel:.2f} pixels")
            except ValueError:
                QtWidgets.QMessageBox.warning(self.controls, "Input Error", "Please enter a valid scale length.")
        elif line_type == "measure_length":
            self.measured_length = pixel_length
            self.update_animal_size()
        elif line_type == "measure_width":
            self.measured_width = pixel_length
            self.update_animal_size()
        elif line_type == "measure_diameter_arena":
            self.edit_arena_diameter.setText(f"{pixel_length:.1f}")
        elif line_type == "measure_diameter_stim":
            self.edit_stim_diameter.setText(f"{pixel_length:.1f}")

    def update_animal_size(self):
        text = []
        if self.measured_length:
            text.append(f"L: {self.measured_length:.1f}px")
        if self.measured_width:
            text.append(f"W: {self.measured_width:.1f}px")
        if self.measured_length and self.measured_width:
            area = self.measured_length * self.measured_width
            text.append(f"Area: ~{area:.0f} px²")
            if self.spin_min_area.value() == 100:
                self.spin_min_area.setValue(max(1, int(area * 0.3)))
            if self.spin_max_area.value() == 10000:
                self.spin_max_area.setValue(int(area * 3.0))
        self.lbl_animal_size.setText(" | ".join(text) if text else "Animal Size: Not Set")

    def update_shape_lists(self):
        self.canvas.viewer.get_all_shapes_data()

    def measure_for_autodetect(self, category):
        self.canvas.viewer.start_drawing(f"measure_diameter_{category}", "measure")

    def autodetect_shapes(self, category, shape_type):
        if "first_frame" not in self.settings:
            QtWidgets.QMessageBox.warning(self.controls, "Error", "Load a video first.")
            return
        if category == "arena":
            diameter_str = self.edit_arena_diameter.text()
            threshold_val = self.spin_arena_threshold.value()
        else:
            diameter_str = self.edit_stim_diameter.text()
            threshold_val = self.spin_stim_threshold.value()
        try:
            expected_size = float(diameter_str)
            if expected_size <= 0:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(self.controls, "Input Error", "Please enter a valid positive diameter/diagonal.")
            return

        frame = self.settings["first_frame"]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        found_count = 0

        if shape_type == "circle":
            expected_radius = expected_size / 2.0
            radius_tolerance = 0.20
            min_rad = int(expected_radius * (1 - radius_tolerance))
            max_rad = int(expected_radius * (1 + radius_tolerance))
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                1,
                min_rad * 2,
                param1=50,
                param2=threshold_val,
                minRadius=min_rad,
                maxRadius=max_rad,
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for c in circles[0, :]:
                    data = {"shape": "circle", "category": category, "geom": [int(c[0]), int(c[1]), int(c[2])]} 
                    self.canvas.viewer.add_shape_item(data)
                    found_count += 1
        elif shape_type == "rect":
            size_tolerance = 0.25
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    diagonal = math.sqrt(w ** 2 + h ** 2)
                    if abs(diagonal - expected_size) < expected_size * size_tolerance:
                        data = {"shape": "rect", "category": category, "geom": [x, y, w, h]}
                        self.canvas.viewer.add_shape_item(data)
                        found_count += 1

        if found_count > 0:
            self.canvas.viewer.shapes_updated.emit()
            QtWidgets.QMessageBox.information(self.controls, "Success", f"Found and added {found_count} {shape_type}(s).")
        else:
            QtWidgets.QMessageBox.warning(self.controls, "Not Found", f"No {shape_type}s found with the specified parameters.")

    def run_analysis(self):
        if "video_path" not in self.settings:
            QtWidgets.QMessageBox.warning(self.controls, "Warning", "Please load a video first.")
            return
        if "pixel_to_mm" not in self.settings:
            QtWidgets.QMessageBox.warning(self.controls, "Warning", "Please set the scale first.")
            return
        dict_path = self.edit_dict_path.text().strip()
        if not dict_path:
            QtWidgets.QMessageBox.warning(self.controls, "Warning", "Please select the ArUco dictionary file.")
            return
        bg_path = self.background_path()
        if bg_path is None or not bg_path.exists():
            QtWidgets.QMessageBox.warning(self.controls, "Warning", "Please build or provide background.png first.")
            return

        all_shapes = self.canvas.viewer.get_all_shapes_data()
        arenas = [s for s in all_shapes if s["category"] == "arena"]
        if not arenas:
            QtWidgets.QMessageBox.warning(self.controls, "Warning", "Please define at least one arena.")
            return

        self.settings["arenas"] = arenas
        self.settings["stimulus_areas"] = [s for s in all_shapes if s["category"] == "stim"]
        self.settings["exp_name"] = self.edit_exp_name.text().strip() or today_experiment_name()
        self.settings["output_dir"] = self.edit_output_dir.text().strip()
        self.settings["start_time_s"] = parse_time_str(self.edit_start_time.text())
        self.settings["end_time_s"] = parse_time_str(self.edit_end_time.text())
        self.settings["min_area"] = self.spin_min_area.value()
        self.settings["max_area"] = self.spin_max_area.value()
        self.follower_settings["min_area"] = float(self.spin_min_area.value())
        self.follower_settings["max_area"] = float(self.spin_max_area.value())
        self.settings["aruco_dict_path"] = dict_path
        self.settings["background_path"] = str(bg_path)
        self.settings["tracking_method"] = "Background Subtraction"
        self.settings["draw_raw_img"] = self.chk_draw_raw_img.isChecked()
        self.settings["draw_filtered_img"] = self.chk_draw_filtered_img.isChecked()
        self.settings["bee_detector_settings"] = dict(self.detector_settings)
        self.settings["bee_follower_settings"] = dict(self.follower_settings)

        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.worker = AnalysisWorker(dict(self.settings))
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.result_ready.connect(self.on_analysis_finished)
        self.worker.start()

    def on_analysis_finished(self, output, error):
        if self.worker is not None:
            self.worker.wait()
        self.btn_run.setEnabled(True)
        if error:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self.controls, "Analysis Error", str(error))
            return
        output_dir = self.save_output_files(output)
        QtWidgets.QMessageBox.information(self.controls, "Finished", f"All output files have been saved to:\n{output_dir}")

    def save_output_files(self, output_data):
        base_name = self.settings["exp_name"]
        output_dir = self.settings.get("output_dir") or os.path.dirname(self.settings["video_path"])
        output_dir = os.path.join(output_dir, f"{base_name}_output")
        os.makedirs(output_dir, exist_ok=True)

        meta_file_path = os.path.join(output_dir, f"{base_name}_info.txt")
        engine = output_data.get("engine")
        marker_ids = list(output_data.get("marker_ids", []))

        with open(meta_file_path, "w", encoding="utf-8") as f:
            f.write("# Video Info\n")
            f.write(f"video_path\t{self.settings['video_path']}\n")
            f.write(f"fps\t{self.settings.get('fps', 30.0):.4f}\n")
            if engine:
                f.write(f"frame_width\t{engine.frame_width}\n")
                f.write(f"frame_height\t{engine.frame_height}\n")
            f.write(f"start_time_str\t{self.edit_start_time.text()}\n")
            f.write(f"end_time_str\t{self.edit_end_time.text()}\n\n")

            f.write("# Inputs\n")
            f.write(f"aruco_dict_path\t{self.settings.get('aruco_dict_path', '')}\n")
            f.write(f"background_path\t{self.settings.get('background_path', '')}\n")
            f.write("tracking_method\tBackground Subtraction\n\n")

            f.write("# Marker IDs\n")
            f.write("marker_ids\t" + ",".join(str(x) for x in marker_ids) + "\n")
            f.write("raw_missing_rule\t-1,-1,-1 means no real detection on that frame\n")
            f.write("filtered_prediction_rule\tIf raw is missing but filtered exists, that point was generated by Kalman prediction\n\n")

            f.write("# Scale\n")
            if "pixel_to_mm" in self.settings:
                f.write(f"pixel_to_mm\t{self.settings['pixel_to_mm']:.8f}\n")
            else:
                f.write("pixel_to_mm\t\n")
            f.write("\n")

            f.write("# GUI Thresholds\n")
            f.write(f"animal_min_area\t{self.settings.get('min_area', '')}\n")
            f.write(f"animal_max_area\t{self.settings.get('max_area', '')}\n\n")

            f.write("# BeeDetector Settings\n")
            for key, value in self.settings.get("bee_detector_settings", {}).items():
                f.write(f"{key}\t{value}\n")
            f.write("\n")

            f.write("# BeeFollower Settings\n")
            for key, value in self.settings.get("bee_follower_settings", {}).items():
                f.write(f"{key}\t{value}\n")
            f.write("\n")

            f.write("# Kalman Settings\n")
            if engine:
                f.write(f"kalman_max_missed_frames\t{engine.max_missed_frames}\n")
                f.write(f"kalman_process_noise\t{engine.kalman_process_noise}\n")
                f.write(f"kalman_measurement_noise\t{engine.kalman_measurement_noise}\n")
                f.write(f"dilate_kernel_size\t{engine.dilate_kernel_size}\n")
                f.write(f"dilate_iterations\t{engine.dilate_iterations}\n")
            f.write("\n")

            f.write("# Arenas\n")
            f.write("id\tshape\tcategory\tgeom\n")
            for shape in self.settings.get("arenas", []):
                f.write(f"{shape['id']}\t{shape['shape']}\t{shape['category']}\t{shape['geom']}\n")
            f.write("\n# Stimulus Areas\n")
            f.write("id\tshape\tcategory\tgeom\n")
            for shape in self.settings.get("stimulus_areas", []):
                f.write(f"{shape['id']}\t{shape['shape']}\t{shape['category']}\t{shape['geom']}\n")

        marker_ids = list(output_data.get("marker_ids", []))
        raw_coords = output_data.get("raw_coords", [])
        filtered_coords = output_data.get("filtered_coords", [])
        raw_angles = output_data.get("raw_angles", [])
        filtered_angles = output_data.get("filtered_angles", [])

        header = "\t".join(
            [f"ID_{marker_id}_X\tID_{marker_id}_Y\tID_{marker_id}_Ang" for marker_id in marker_ids]
        )

        # Vektörize edildi (eski hali: her frame x her marker için Python
        # döngüsüyle string oluşturuyordu - 8 saatlik video x çok sayıda
        # marker'da bu adım dakikalar sürebiliyordu).
        def _coords_to_matrix(coords_history, angles_history, n_markers):
            n_frames = len(coords_history)
            if n_frames == 0:
                return np.empty((0, n_markers * 3))
            pos_arr = np.asarray(coords_history, dtype=float)  # (n_frames, n_markers, 2)
            ang_arr = np.asarray(angles_history, dtype=float)  # (n_frames, n_markers)
            out = np.empty((n_frames, n_markers * 3), dtype=float)
            out[:, 0::3] = pos_arr[:, :, 0]
            out[:, 1::3] = pos_arr[:, :, 1]
            out[:, 2::3] = ang_arr
            return out

        n_markers = len(marker_ids)
        raw_matrix = _coords_to_matrix(raw_coords, raw_angles, n_markers)
        filtered_matrix = _coords_to_matrix(filtered_coords, filtered_angles, n_markers)

        # X/Y sütunları tam sayı (piksel) olarak, açı sütunu 6 ondalıkla yazılır - eski çıktı formatıyla birebir aynı.
        fmt = (["%d", "%d", "%.6f"] * n_markers) if n_markers > 0 else []

        np.savetxt(
            os.path.join(output_dir, f"{base_name}_coordinates_raw.txt"),
            raw_matrix,
            delimiter="\t",
            header=header,
            fmt=fmt if fmt else "%s",
            comments="",
        )

        np.savetxt(
            os.path.join(output_dir, f"{base_name}_coordinates_filtered.txt"),
            filtered_matrix,
            delimiter="\t",
            header=header,
            fmt=fmt if fmt else "%s",
            comments="",
        )

        self.save_trajectory_plot(output_data, output_dir)
        if self.chk_save_video.isChecked():
            self.save_track_video(output_data, output_dir)
        return output_dir

    def save_trajectory_plot(self, output_data, output_dir):
        engine = output_data["engine"]
        filtered_coords = output_data["filtered_coords"]
        raw_coords = output_data["raw_coords"]
        marker_ids = list(output_data.get("marker_ids", []))

        cap = cv2.VideoCapture(self.settings["video_path"])
        end_frame_idx = int(parse_time_str(self.edit_end_time.text()) * self.settings["fps"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame_idx)
        ret, last_frame = cap.read()
        cap.release()
        background = last_frame if ret else (engine.last_processed_frame if engine.last_processed_frame is not None else self.settings["first_frame"])

        from matplotlib.patches import Circle, Rectangle, Polygon

        all_shapes_for_plot = self.settings["arenas"] + self.settings["stimulus_areas"]
        draw_raw = self.settings.get("draw_raw_img", True)
        draw_filtered = self.settings.get("draw_filtered_img", True)

        cmap = plt.get_cmap("hsv", max(len(marker_ids), 1) + 1)

        for marker_index, marker_id in enumerate(marker_ids):
            fig, ax = plt.subplots(figsize=(engine.frame_width / 100, engine.frame_height / 100), dpi=200)
            ax.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
            ax.set_xlim(0, engine.frame_width)
            ax.set_ylim(engine.frame_height, 0)
            ax.set_aspect("equal")
            ax.axis("off")

            for shape in all_shapes_for_plot:
                color = "cyan" if shape["category"] == "arena" else "yellow"
                lw = 1.5
                if shape["shape"] == "circle":
                    ax.add_patch(Circle((shape["geom"][0], shape["geom"][1]), shape["geom"][2], fill=False, ec=color, lw=lw))
                elif shape["shape"] == "rect":
                    ax.add_patch(Rectangle((shape["geom"][0], shape["geom"][1]), shape["geom"][2], shape["geom"][3], fill=False, ec=color, lw=lw))
                elif shape["shape"] == "poly":
                    ax.add_patch(Polygon(np.array(shape["geom"]), fill=False, ec=color, lw=lw))
                center = get_shape_center(shape)
                ax.text(
                    center[0], center[1], shape["id"], color="white", ha="center", va="center", fontsize=10,
                    weight="bold", bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.2")
                )

            line_color = cmap(marker_index)

            if draw_raw:
                raw_points = np.array([frame[marker_index] for frame in raw_coords if frame[marker_index][0] != -1], dtype=float)
                if len(raw_points) > 1:
                    ax.plot(raw_points[:, 0], raw_points[:, 1], color=line_color, linewidth=1.0, alpha=0.35)

            if draw_filtered:
                points_segments = []
                current_segment = []
                for pos in [frame[marker_index] for frame in filtered_coords]:
                    if pos[0] != -1:
                        current_segment.append(pos)
                    else:
                        if len(current_segment) > 1:
                            points_segments.append(np.array(current_segment, dtype=float))
                        current_segment = []
                if len(current_segment) > 1:
                    points_segments.append(np.array(current_segment, dtype=float))
                for segment in points_segments:
                    ax.plot(segment[:, 0], segment[:, 1], color=line_color, linewidth=1.8)

            valid_filtered = [frame[marker_index] for frame in filtered_coords if frame[marker_index][0] != -1]
            if valid_filtered:
                last_point = valid_filtered[-1]
                ax.text(
                    last_point[0] + 8,
                    last_point[1] + 8,
                    f"ID {marker_id}",
                    color="white",
                    fontsize=10,
                    weight="bold",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.2"),
                )

            plt.savefig(os.path.join(output_dir, f"{self.settings['exp_name']}_trajectory_id_{marker_id}.png"), bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    def save_track_video(self, output_data, output_dir):
        engine = output_data["engine"]
        raw_coords = output_data["raw_coords"]
        filtered_coords = output_data["filtered_coords"]
        raw_angles = output_data.get("raw_angles", [])
        filtered_angles = output_data.get("filtered_angles", [])
        filtered_is_prediction = output_data.get("filtered_is_prediction", [])
        marker_ids = list(output_data.get("marker_ids", []))

        writer = cv2.VideoWriter(
            os.path.join(output_dir, f"{self.settings['exp_name']}_track.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            engine.fps,
            (engine.frame_width, engine.frame_height),
        )
        cap = cv2.VideoCapture(self.settings["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, engine.start_frame)

        cmap = plt.get_cmap("hsv", max(len(marker_ids), 1) + 1)

        for frame_index, (frame_raw, frame_filtered) in enumerate(zip(raw_coords, filtered_coords)):
            ret, frame = cap.read()
            if not ret:
                break

            for shape in self.settings["arenas"] + self.settings["stimulus_areas"]:
                color = (255, 255, 0) if shape["category"] == "arena" else (0, 255, 255)
                if shape["shape"] == "circle":
                    cv2.circle(frame, (shape["geom"][0], shape["geom"][1]), shape["geom"][2], color, 2)
                elif shape["shape"] == "rect":
                    cv2.rectangle(frame, (shape["geom"][0], shape["geom"][1]), (shape["geom"][0] + shape["geom"][2], shape["geom"][1] + shape["geom"][3]), color, 2)
                elif shape["shape"] == "poly":
                    cv2.polylines(frame, [np.array(shape["geom"], dtype=np.int32)], True, color, 2)

            for i, marker_id in enumerate(marker_ids):
                raw_pos = frame_raw[i]
                filtered_pos = frame_filtered[i]
                raw_ang = raw_angles[frame_index][i] if frame_index < len(raw_angles) else -1.0
                filtered_ang = filtered_angles[frame_index][i] if frame_index < len(filtered_angles) else -1.0
                is_pred = bool(filtered_is_prediction[frame_index][i]) if frame_index < len(filtered_is_prediction) else False

                rgb = cmap(i)[:3]
                bgr = tuple(int(round(255 * c)) for c in (rgb[2], rgb[1], rgb[0]))

                if raw_pos[0] != -1:
                    cv2.circle(frame, (raw_pos[0], raw_pos[1]), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID {marker_id} raw {raw_ang:.1f}", (raw_pos[0] + 6, raw_pos[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

                if filtered_pos[0] != -1:
                    draw_color = (255, 0, 255) if is_pred else bgr
                    cv2.circle(frame, (filtered_pos[0], filtered_pos[1]), 6, draw_color, 2)
                    label = f"ID {marker_id} {'pred' if is_pred else 'filt'} {filtered_ang:.1f}"
                    cv2.putText(frame, label, (filtered_pos[0] + 6, filtered_pos[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 1, cv2.LINE_AA)

            writer.write(frame)

        writer.release()
        cap.release()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    _ = MainWindow()
    sys.exit(app.exec())
