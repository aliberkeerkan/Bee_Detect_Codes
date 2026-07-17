from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

REQUIRED = [
    ("PySide6", "PySide6"),
    ("pandas", "pandas"),
    ("cv2", "opencv-python"),
    ("matplotlib", "matplotlib"),
    ("plotly", "plotly"),
]


def _ensure(mod_name: str, pip_name: str) -> bool:
    try:
        importlib.import_module(mod_name)
        return True
    except ImportError:
        try:
            print(f"Installing missing package: {pip_name}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--user", pip_name], check=True)
            importlib.import_module(mod_name)
            return True
        except Exception as exc:
            print(f"FATAL: Missing package '{pip_name}'. Please install it manually.")
            print(exc)
            return False


_missing = [pip_name for mod_name, pip_name in REQUIRED if not _ensure(mod_name, pip_name)]
if _missing:
    sys.exit(1)

try:
    import analysis_core as analysis_core
except ImportError:
    local_core = Path(__file__).with_name("analysis_core_v5.py")
    if not local_core.exists():
        print("FATAL: Could not find analysis_core_v5.py in the same directory.")
        sys.exit(1)
    sys.path.insert(0, str(local_core.parent))
    import analysis_core as analysis_core

from PySide6 import QtCore, QtWidgets


class AnalysisWorker(QtCore.QThread):
    """Runs build_analysis_bundle off the GUI thread.

    Without this, the whole window freezes for the duration of the analysis
    (which can be many minutes on large datasets) and the OS may report the
    app as "Not Responding" and kill it before it ever gets a chance to finish.
    """

    progress = QtCore.Signal(str)
    # Named result_ready (not "finished") to avoid shadowing QThread's own
    # built-in `finished` signal, which Qt relies on for thread lifecycle
    # bookkeeping - reusing the name caused "QThread: Destroyed while thread
    # is still running" aborts on shutdown.
    result_ready = QtCore.Signal(object, object)  # (bundle_or_None, error_or_None)

    def __init__(self, kwargs: dict) -> None:
        super().__init__()
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            bundle = analysis_core.build_analysis_bundle(
                progress_callback=self.progress.emit,
                **self._kwargs,
            )
            self.result_ready.emit(bundle, None)
        except Exception as exc:  # noqa: BLE001 - report any failure back to the GUI thread
            self.result_ready.emit(None, exc)


class AnalysisWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tracking Data Analyzer v5")
        self.resize(760, 560)

        self.output_folder_path: str | None = None
        self.experiment_name: str | None = None
        self.discovered_files: dict | None = None
        self.info_data: dict | None = None
        self.bundle: analysis_core.AnalysisBundle | None = None
        self.last_export_directory: str | None = None
        self._worker: AnalysisWorker | None = None

        self._build_ui()
        self._set_initial_state()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        folder_layout = QtWidgets.QHBoxLayout()
        self.lbl_folder = QtWidgets.QLabel("No tracking folder selected.")
        self.lbl_folder.setWordWrap(True)
        self.btn_select_folder = QtWidgets.QPushButton("Load Tracking Folder...")
        folder_layout.addWidget(self.lbl_folder, 1)
        folder_layout.addWidget(self.btn_select_folder)
        root.addLayout(folder_layout)

        self.lbl_info = QtWidgets.QLabel("Select a folder that contains one *_info.txt file and matching raw/filtered coordinate files.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-style: italic;")
        root.addWidget(self.lbl_info)

        options_group = QtWidgets.QGroupBox("Analysis Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)

        bin_layout = QtWidgets.QHBoxLayout()
        bin_layout.addWidget(QtWidgets.QLabel("Time bin (seconds):"))
        self.spin_time_bin = QtWidgets.QSpinBox(minimum=1, maximum=24 * 3600, value=30)
        bin_layout.addWidget(self.spin_time_bin)
        bin_layout.addStretch(1)
        options_layout.addLayout(bin_layout)

        metrics_group = QtWidgets.QGroupBox("Metrics")
        metrics_layout = QtWidgets.QVBoxLayout(metrics_group)
        self.chk_metric_speed = QtWidgets.QCheckBox("Average Speed (mm/s)")
        self.chk_metric_dist_arena = QtWidgets.QCheckBox("Average Distance to Arena Center")
        self.chk_metric_stim_duration = QtWidgets.QCheckBox("Duration in Stimulus Area")
        self.chk_metric_stim_entries = QtWidgets.QCheckBox("Entries / Exits from Stimulus Area")
        self.chk_metric_stim_dist = QtWidgets.QCheckBox("Average Distance to Stimulus Center")
        for chk in [
            self.chk_metric_speed,
            self.chk_metric_dist_arena,
            self.chk_metric_stim_duration,
            self.chk_metric_stim_entries,
            self.chk_metric_stim_dist,
        ]:
            chk.setChecked(True)
            metrics_layout.addWidget(chk)
        options_layout.addWidget(metrics_group)

        root.addWidget(options_group)

        self.txt_shapes = QtWidgets.QPlainTextEdit()
        self.txt_shapes.setReadOnly(True)
        self.txt_shapes.setPlaceholderText("Arena / stimulus summary will appear here after loading a folder.")
        root.addWidget(self.txt_shapes, 1)

        self.lbl_status = QtWidgets.QLabel("")
        root.addWidget(self.lbl_status)

        button_layout = QtWidgets.QHBoxLayout()
        self.btn_run_analysis = QtWidgets.QPushButton("Run Analysis")
        self.btn_save_tables = QtWidgets.QPushButton("Save Tables (.tsv)...")
        self.btn_save_plots = QtWidgets.QPushButton("Save Plots (.png)...")
        self.btn_save_report = QtWidgets.QPushButton("Save Report (HTML/CSS)...")
        self.btn_exit = QtWidgets.QPushButton("Exit")
        button_layout.addWidget(self.btn_run_analysis)
        button_layout.addWidget(self.btn_save_tables)
        button_layout.addWidget(self.btn_save_plots)
        button_layout.addWidget(self.btn_save_report)
        button_layout.addStretch(1)
        button_layout.addWidget(self.btn_exit)
        root.addLayout(button_layout)

        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_run_analysis.clicked.connect(self.run_analysis)
        self.btn_save_tables.clicked.connect(self.save_tables)
        self.btn_save_plots.clicked.connect(self.save_plots)
        self.btn_save_report.clicked.connect(self.save_report)
        self.btn_exit.clicked.connect(self.close)

    def _set_initial_state(self) -> None:
        self.btn_run_analysis.setEnabled(False)
        self.btn_save_tables.setEnabled(False)
        self.btn_save_plots.setEnabled(False)
        self.btn_save_report.setEnabled(False)

    def _selected_metrics(self) -> dict:
        return {
            "speed": self.chk_metric_speed.isChecked(),
            "dist_arena": self.chk_metric_dist_arena.isChecked(),
            "stim_duration": self.chk_metric_stim_duration.isChecked(),
            "stim_entries": self.chk_metric_stim_entries.isChecked(),
            "stim_dist": self.chk_metric_stim_dist.isChecked(),
        }

    def select_folder(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Tracking Output Folder")
        if not directory:
            return

        self.output_folder_path = directory
        self.lbl_folder.setText(directory)
        self.lbl_folder.setToolTip(directory)
        self.bundle = None
        self.btn_save_tables.setEnabled(False)
        self.btn_save_plots.setEnabled(False)
        self.btn_save_report.setEnabled(False)

        try:
            discovered = analysis_core.discover_experiment_files(directory)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Folder Error", str(exc))
            self.lbl_status.setText("Could not determine experiment files.")
            self.btn_run_analysis.setEnabled(False)
            return

        if not discovered:
            QtWidgets.QMessageBox.warning(self, "Files Not Found", "Could not find any *_info.txt file in the selected folder.")
            self.lbl_status.setText("No experiment info file found.")
            self.btn_run_analysis.setEnabled(False)
            return

        self.discovered_files = discovered
        self.experiment_name = discovered["experiment_name"]
        self.info_data = analysis_core.parse_info_file(discovered["info_path"])
        if self.info_data is None:
            QtWidgets.QMessageBox.critical(self, "Parse Error", f"Failed to parse info file:\n{discovered['info_path']}")
            self.lbl_status.setText("Failed to parse info file.")
            self.btn_run_analysis.setEnabled(False)
            return

        raw_available = bool(discovered.get("raw_path"))
        filtered_available = bool(discovered.get("filtered_path"))
        if not raw_available and not filtered_available:
            QtWidgets.QMessageBox.warning(self, "Files Not Found", "No raw or filtered coordinate files were found for this experiment.")
            self.lbl_status.setText("No coordinate files found.")
            self.btn_run_analysis.setEnabled(False)
            return

        fps = self.info_data.get("fps", "N/A")
        pixel_to_mm = self.info_data.get("pixel_to_mm", 1.0)
        info_text = (
            f"Experiment: {self.experiment_name}\n"
            f"FPS: {fps}\n"
            f"Scale: {pixel_to_mm:.6f} mm/px\n"
            f"Frame size: {self.info_data.get('frame_width')} x {self.info_data.get('frame_height')}\n"
            f"Raw available: {'Yes' if raw_available else 'No'}\n"
            f"Filtered available: {'Yes' if filtered_available else 'No'}\n"
            f"Arenas: {', '.join(self.info_data.get('arenas', {}).keys()) or 'None'}\n"
            f"Stimuli: {', '.join(self.info_data.get('stimulus_areas', {}).keys()) or 'None'}\n"
            f"Arena/Stim intersections: {analysis_core.shape_intersections(self.info_data)}"
        )
        self.txt_shapes.setPlainText(info_text)
        self.lbl_info.setText("Ready. Analysis will build framewise + time-binned tables, static plots, and an interactive HTML report.")
        self.lbl_status.setText(f"Loaded experiment: {self.experiment_name}")
        self.btn_run_analysis.setEnabled(True)

    def _suggest_export_directory(self) -> str:
        base_dir = self.output_folder_path or os.getcwd()
        if self.last_export_directory and os.path.isdir(self.last_export_directory):
            return self.last_export_directory
        if self.experiment_name:
            return os.path.join(base_dir, f"{self.experiment_name}_analysis_output")
        return base_dir

    def run_analysis(self) -> None:
        if not self.discovered_files or not self.info_data or not self.experiment_name:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please load a tracking folder first.")
            return

        self.lbl_status.setText("Loading tracking files...")
        QtWidgets.QApplication.processEvents()

        try:
            raw_df = analysis_core.load_coordinate_file(self.discovered_files.get("raw_path"))
            filtered_df = analysis_core.load_coordinate_file(self.discovered_files.get("filtered_path"))
            if raw_df is None and filtered_df is None:
                raise ValueError("No usable coordinate file could be loaded.")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Analysis Error", f"An error occurred while loading coordinate files:\n{exc}")
            self.lbl_status.setText("Failed to load coordinate files.")
            return

        self.btn_run_analysis.setEnabled(False)
        self.btn_select_folder.setEnabled(False)
        self.btn_save_tables.setEnabled(False)
        self.btn_save_plots.setEnabled(False)
        self.btn_save_report.setEnabled(False)
        self.lbl_status.setText("Building framewise and binned datasets...")

        worker_kwargs = dict(
            experiment_name=self.experiment_name,
            info_data=self.info_data,
            raw_df=raw_df,
            filtered_df=filtered_df,
            bin_seconds=self.spin_time_bin.value(),
            selected_metrics=self._selected_metrics(),
            source_paths={
                "raw": self.discovered_files.get("raw_path"),
                "filtered": self.discovered_files.get("filtered_path"),
            },
        )

        self._worker = AnalysisWorker(worker_kwargs)
        self._worker.progress.connect(self.lbl_status.setText)
        self._worker.result_ready.connect(self._on_analysis_finished)
        self._worker.start()

    def _on_analysis_finished(self, bundle, error) -> None:
        if self._worker is not None:
            self._worker.wait()
        self.btn_run_analysis.setEnabled(True)
        self.btn_select_folder.setEnabled(True)

        if error is not None:
            import traceback
            traceback.print_exception(type(error), error, error.__traceback__)
            QtWidgets.QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis:\n{error}")
            self.lbl_status.setText("Analysis failed.")
            self.bundle = None
            self.btn_save_tables.setEnabled(False)
            self.btn_save_plots.setEnabled(False)
            self.btn_save_report.setEnabled(False)
            return

        self.bundle = bundle
        bee_count = len(self.bundle.bee_ids)
        compare_rows = len(self.bundle.framewise_compare_df)
        self.lbl_status.setText(f"Analysis complete. Bees: {bee_count} | Framewise rows: {compare_rows}")
        self.btn_save_tables.setEnabled(True)
        self.btn_save_plots.setEnabled(True)
        self.btn_save_report.setEnabled(True)

    def _ask_output_directory(self, title: str) -> str | None:
        suggested = self._suggest_export_directory()
        os.makedirs(suggested, exist_ok=True)
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, title, suggested)
        if selected:
            self.last_export_directory = selected
        return selected or None

    def save_tables(self) -> None:
        if not self.bundle:
            QtWidgets.QMessageBox.warning(self, "No Results", "Run the analysis first.")
            return
        output_dir = self._ask_output_directory("Select Folder to Save TSV Tables")
        if not output_dir:
            return
        try:
            self.lbl_status.setText("Saving tables...")
            QtWidgets.QApplication.processEvents()
            saved = analysis_core.export_tables(self.bundle, output_dir)
            self.lbl_status.setText("Tables saved successfully.")
            QtWidgets.QMessageBox.information(self, "Tables Saved", "Saved tables:\n\n" + "\n".join(saved.values()))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not save tables:\n{exc}")
            self.lbl_status.setText("Error saving tables.")

    def save_plots(self) -> None:
        if not self.bundle:
            QtWidgets.QMessageBox.warning(self, "No Results", "Run the analysis first.")
            return
        output_dir = self._ask_output_directory("Select Folder to Save Static Plots")
        if not output_dir:
            return
        try:
            self.lbl_status.setText("Generating PNG plots and heatmaps...")
            QtWidgets.QApplication.processEvents()
            saved = analysis_core.create_static_plots(self.bundle, output_dir)
            self.lbl_status.setText("Plots saved successfully.")
            QtWidgets.QMessageBox.information(self, "Plots Saved", f"Saved {len(saved)} PNG files under:\n{os.path.join(output_dir, 'plots')}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Plot Error", f"Could not save plots:\n{exc}")
            self.lbl_status.setText("Error saving plots.")

    def save_report(self) -> None:
        if not self.bundle:
            QtWidgets.QMessageBox.warning(self, "No Results", "Run the analysis first.")
            return
        output_dir = self._ask_output_directory("Select Folder to Save HTML Report")
        if not output_dir:
            return
        try:
            self.lbl_status.setText("Exporting interactive HTML report...")
            QtWidgets.QApplication.processEvents()
            saved = analysis_core.export_html_report(self.bundle, output_dir)
            self.lbl_status.setText("Report saved successfully.")
            QtWidgets.QMessageBox.information(self, "Report Saved", "Saved report files:\n\n" + "\n".join(saved.values()))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Report Error", f"Could not save report:\n{exc}")
            self.lbl_status.setText("Error saving report.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AnalysisWindow()
    window.show()
    sys.exit(app.exec())
