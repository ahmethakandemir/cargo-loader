"""Main application window – 3-D viewer + side control panel."""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QLabel, QPushButton, QProgressBar, QTableWidget,
    QTableWidgetItem, QSpinBox, QDoubleSpinBox, QHeaderView,
    QAbstractItemView, QStatusBar, QDialog, QFormLayout,
    QLineEdit, QCheckBox, QDialogButtonBox, QMessageBox,
    QColorDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont

from .models import Container, Box
from .genetic import GeneticAlgorithm
from .annealing import SimulatedAnnealing
from .viewer import ContainerViewer, OPENGL_OK
from .test_data import create_test_boxes, BOX_COLORS


# ── Add-box dialog ────────────────────────────────────────────────

class AddBoxDialog(QDialog):
    """Modal dialog for adding a new box to the list."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Box")
        self.setMinimumWidth(320)

        form = QFormLayout(self)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Crate X")
        form.addRow("Name:", self.name_edit)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 5000)
        self.width_spin.setValue(60)
        self.width_spin.setSuffix("  (x)")
        form.addRow("Width:", self.width_spin)

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 5000)
        self.depth_spin.setValue(60)
        self.depth_spin.setSuffix("  (z)")
        form.addRow("Depth:", self.depth_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 5000)
        self.height_spin.setValue(60)
        self.height_spin.setSuffix("  (y)")
        form.addRow("Height:", self.height_spin)

        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0.1, 99999.0)
        self.weight_spin.setValue(10.0)
        self.weight_spin.setDecimals(1)
        self.weight_spin.setSuffix(" kg")
        form.addRow("Weight:", self.weight_spin)

        self.rotate_check = QCheckBox("Allow 6 orientations")
        self.rotate_check.setChecked(True)
        form.addRow("Can Rotate:", self.rotate_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


# ── Box info / edit dialog ────────────────────────────────────────

class BoxInfoDialog(QDialog):
    """Modal dialog shown when the user clicks a placed box in the 3-D view.

    Editable fields : name, colour.
    Read-only fields: ID, position, placed size, original size, weight,
                      can-rotate flag.
    """

    def __init__(self, placed_box, parent=None):
        super().__init__(parent)
        self._pb = placed_box
        self._box = placed_box.box
        self._color = tuple(self._box.color)   # (r, g, b) in 0-1

        self.setWindowTitle(f"Box Info — {self._box.name}")
        self.setMinimumWidth(370)

        form = QFormLayout(self)

        # ── Editable: name ──────────────────────────────────────
        self.name_edit = QLineEdit(self._box.name)
        form.addRow("Name:", self.name_edit)

        # ── Read-only fields ────────────────────────────────────
        form.addRow("Box ID:", QLabel(str(self._box.id)))

        pos_text = (f"x = {placed_box.x:.1f},  "
                    f"y = {placed_box.y:.1f},  "
                    f"z = {placed_box.z:.1f}")
        form.addRow("Position:", QLabel(pos_text))

        placed_dims = (f"{placed_box.width:.0f}  ×  "
                       f"{placed_box.depth:.0f}  ×  "
                       f"{placed_box.height:.0f}")
        form.addRow("Placed Size (W×D×H):", QLabel(placed_dims))

        orig_dims = (f"{self._box.width:.0f}  ×  "
                     f"{self._box.depth:.0f}  ×  "
                     f"{self._box.height:.0f}")
        form.addRow("Original Size (W×D×H):", QLabel(orig_dims))

        form.addRow("Weight:", QLabel(f"{self._box.weight:.1f} kg"))
        form.addRow("Can Rotate:",
                     QLabel("Yes  (6 orientations)"
                            if self._box.can_rotate else "No"))

        # ── Editable: colour ────────────────────────────────────
        self._color_btn = QPushButton()
        self._update_color_btn()
        self._color_btn.clicked.connect(self._pick_color)
        form.addRow("Colour:", self._color_btn)

        # ── OK / Cancel ─────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    # helpers ---------------------------------------------------------

    def _update_color_btn(self):
        r, g, b = (int(c * 255) for c in self._color)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        txt = "black" if lum > 128 else "white"
        self._color_btn.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r},{g},{b}); "
            f"color: {txt}; min-width: 90px; min-height: 26px; "
            f"border: 1px solid #666; border-radius: 3px; }}")
        self._color_btn.setText(f"  ({r}, {g}, {b})  ")

    def _pick_color(self):
        r, g, b = (int(c * 255) for c in self._color)
        chosen = QColorDialog.getColor(
            QColor(r, g, b), self, "Choose Box Colour")
        if chosen.isValid():
            self._color = (chosen.redF(), chosen.greenF(), chosen.blueF())
            self._update_color_btn()

    # public getters --------------------------------------------------

    def get_name(self) -> str:
        return self.name_edit.text().strip() or self._box.name

    def get_color(self):
        return self._color


# ── Optimiser worker thread (GA + SA) ─────────────────────────────

class _OptWorker(QThread):
    """Runs GA (global search) then SA (local refinement) in sequence."""
    # step, total_steps, fitness, placed|None, phase_label
    progress = pyqtSignal(int, int, float, object, str)
    finished = pyqtSignal(object, float)

    def __init__(self, container, boxes, pop_size, generations, sa_iters):
        super().__init__()
        self.container = container
        self.boxes = boxes
        self.pop_size = pop_size
        self.generations = generations
        self.sa_iters = sa_iters
        self._ga = None
        self._sa = None
        self._stop_requested = False

    def run(self):
        total_steps = self.generations + self.sa_iters

        # ── Phase 1: Genetic Algorithm ───────────────────────────
        self._ga = GeneticAlgorithm(
            self.container, self.boxes,
            pop_size=self.pop_size,
            generations=self.generations,
        )

        def _ga_cb(gen, _total_gens, fitness, placed):
            label = f"GA  gen {gen + 1} / {self.generations}"
            self.progress.emit(gen, total_steps, fitness, placed, label)

        self._ga.on_progress = _ga_cb
        ga_placed, ga_fitness = self._ga.run()

        best_placed = ga_placed
        best_fitness = ga_fitness

        # ── Phase 2: Simulated Annealing (refine GA result) ──────
        if (self.sa_iters > 0
                and not self._stop_requested
                and self._ga.best_individual is not None):
            self._sa = SimulatedAnnealing(
                self.container, self.boxes,
                initial=self._ga.best_individual,
                iterations=self.sa_iters,
            )

            def _sa_cb(it, _total_iters, fitness, placed):
                label = f"SA  iter {it + 1} / {self.sa_iters}"
                self.progress.emit(
                    self.generations + it, total_steps,
                    fitness, placed, label,
                )

            self._sa.on_progress = _sa_cb
            sa_placed, sa_fitness = self._sa.run()

            if sa_fitness >= best_fitness:
                best_placed = sa_placed
                best_fitness = sa_fitness

        self.finished.emit(best_placed, best_fitness)

    def stop(self):
        self._stop_requested = True
        if self._ga:
            self._ga.stop()
        if self._sa:
            self._sa.stop()


# ── Main window ───────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CargoLoader — 3D Container Packing with Genetic Algorithm")
        self.setMinimumSize(1100, 650)
        self.resize(1400, 800)

        self.container = Container(250, 500, 250)
        self.boxes = create_test_boxes()
        self.placed_boxes = []
        self._opt_worker = None

        self._build_ui()
        self._populate_box_table()
        self._update_stats()
        self.viewer.set_data(self.container, [])

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left: 3-D viewer (or fallback label)
        if OPENGL_OK:
            self.viewer = ContainerViewer()
            self.viewer.box_clicked.connect(self._on_box_clicked)
        else:
            self.viewer = QLabel("OpenGL not available.\n"
                                 "Install PyOpenGL:\n  pip install PyOpenGL")
            self.viewer.setAlignment(Qt.AlignCenter)
        splitter.addWidget(self.viewer)

        # Right: control panel
        panel = QWidget()
        panel.setMinimumWidth(310)
        panel.setMaximumWidth(430)
        play = QVBoxLayout(panel)
        play.setSpacing(10)

        self._build_container_group(play)
        self._build_ga_group(play)
        self._build_box_table_group(play)
        self._build_stats_group(play)
        play.addStretch()

        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # Status bar
        sb = QStatusBar()
        sb.showMessage("Click box: inspect/edit  |  Left-drag: rotate  |  Right-drag: pan  |  Scroll: zoom")
        self.setStatusBar(sb)

    # -- Container settings ----------------------------------------

    def _build_container_group(self, parent):
        grp = QGroupBox("Container Dimensions")
        lay = QHBoxLayout(grp)

        def _spin(label, value):
            lay.addWidget(QLabel(label))
            sb = QSpinBox()
            sb.setRange(50, 5000)
            sb.setValue(value)
            sb.setSingleStep(10)
            sb.valueChanged.connect(self._on_container_changed)
            lay.addWidget(sb)
            return sb

        self._cw = _spin("W:", int(self.container.width))
        self._cd = _spin("D:", int(self.container.depth))
        self._ch = _spin("H:", int(self.container.height))
        parent.addWidget(grp)

    def _on_container_changed(self):
        self.container = Container(
            float(self._cw.value()),
            float(self._cd.value()),
            float(self._ch.value()),
        )
        if OPENGL_OK:
            self.viewer.set_data(self.container, self.placed_boxes)
        self._update_stats()

    # -- Optimiser controls (GA + SA) ---------------------------------

    def _build_ga_group(self, parent):
        grp = QGroupBox("Optimiser  (GA + SA)")
        lay = QVBoxLayout(grp)

        # GA parameters
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Population:"))
        self._pop_spin = QSpinBox()
        self._pop_spin.setRange(10, 500)
        self._pop_spin.setValue(60)
        row1.addWidget(self._pop_spin)

        row1.addWidget(QLabel("GA Gens:"))
        self._gen_spin = QSpinBox()
        self._gen_spin.setRange(10, 2000)
        self._gen_spin.setValue(100)
        row1.addWidget(self._gen_spin)
        lay.addLayout(row1)

        # SA parameters
        row_sa = QHBoxLayout()
        row_sa.addWidget(QLabel("SA Iterations:"))
        self._sa_spin = QSpinBox()
        self._sa_spin.setRange(0, 50000)
        self._sa_spin.setValue(2000)
        self._sa_spin.setSingleStep(500)
        self._sa_spin.setToolTip("0 = skip SA phase")
        row_sa.addWidget(self._sa_spin)
        lay.addLayout(row_sa)

        # Buttons
        row2 = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run")
        self._run_btn.setMinimumHeight(32)
        self._run_btn.clicked.connect(self._on_run)
        row2.addWidget(self._run_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setMinimumHeight(32)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        row2.addWidget(self._stop_btn)

        self._reset_btn = QPushButton("↺  Reset")
        self._reset_btn.setMinimumHeight(32)
        self._reset_btn.clicked.connect(self._on_reset)
        row2.addWidget(self._reset_btn)
        lay.addLayout(row2)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        lay.addWidget(self._progress)

        self._gen_label = QLabel("Phase: — / —")
        lay.addWidget(self._gen_label)
        self._fit_label = QLabel("Best Fitness: —")
        lay.addWidget(self._fit_label)
        parent.addWidget(grp)

    # -- Box list --------------------------------------------------

    def _build_box_table_group(self, parent):
        grp = QGroupBox("Box List")
        lay = QVBoxLayout(grp)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["", "Name", "W×D×H", "Wt (kg)", "Rot", "Placed"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Fixed)
        hdr.resizeSection(0, 24)
        for c in range(1, 6):
            hdr.setSectionResizeMode(c, QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setMinimumHeight(180)

        lay.addWidget(self._table)

        # Add / Remove buttons
        btn_row = QHBoxLayout()

        self._add_box_btn = QPushButton("+  Add Box")
        self._add_box_btn.clicked.connect(self._on_add_box)
        btn_row.addWidget(self._add_box_btn)

        self._rm_box_btn = QPushButton("-  Remove")
        self._rm_box_btn.clicked.connect(self._on_remove_box)
        btn_row.addWidget(self._rm_box_btn)

        lay.addLayout(btn_row)
        parent.addWidget(grp)

    def _populate_box_table(self):
        placed_ids = {pb.box.id for pb in self.placed_boxes}
        self._table.setRowCount(len(self.boxes))
        for i, box in enumerate(self.boxes):
            # Colour swatch
            item = QTableWidgetItem()
            r, g, b = (int(c * 255) for c in box.color)
            item.setBackground(QColor(r, g, b))
            item.setFlags(Qt.ItemIsEnabled)
            self._table.setItem(i, 0, item)

            self._table.setItem(i, 1, QTableWidgetItem(box.name))
            dims = f"{box.width:.0f}×{box.depth:.0f}×{box.height:.0f}"
            self._table.setItem(i, 2, QTableWidgetItem(dims))
            self._table.setItem(i, 3, QTableWidgetItem(f"{box.weight:.1f}"))
            self._table.setItem(i, 4, QTableWidgetItem("✓" if box.can_rotate else "✗"))
            placed = "✓" if box.id in placed_ids else "—"
            pi = QTableWidgetItem(placed)
            pi.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(i, 5, pi)

    # -- Statistics ------------------------------------------------

    def _build_stats_group(self, parent):
        grp = QGroupBox("Statistics")
        lay = QVBoxLayout(grp)
        bold = QFont()
        bold.setBold(True)

        self._stat_placed = QLabel()
        self._stat_vol    = QLabel()
        self._stat_weight = QLabel()
        self._stat_cvol   = QLabel()

        for w in (self._stat_placed, self._stat_vol,
                  self._stat_weight, self._stat_cvol):
            w.setFont(bold)
            lay.addWidget(w)
        parent.addWidget(grp)

    def _update_stats(self):
        n_placed = len(self.placed_boxes)
        n_total  = len(self.boxes)
        self._stat_placed.setText(f"Boxes placed: {n_placed} / {n_total}")

        cvol = (self.container.width * self.container.depth
                * self.container.height)
        used = sum(p.width * p.depth * p.height for p in self.placed_boxes)
        pct  = used / cvol * 100 if cvol > 0 else 0
        self._stat_vol.setText(f"Volume used:  {pct:.1f}%  "
                               f"({used:,.0f} / {cvol:,.0f})")

        wt = sum(p.box.weight for p in self.placed_boxes)
        self._stat_weight.setText(f"Total weight: {wt:.1f} kg")
        self._stat_cvol.setText(
            f"Container:    {self.container.width:.0f} × "
            f"{self.container.depth:.0f} × {self.container.height:.0f}")

    # ── Optimiser callbacks ──────────────────────────────────────

    def _on_run(self):
        self.container = Container(
            float(self._cw.value()),
            float(self._cd.value()),
            float(self._ch.value()),
        )
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setValue(0)

        self._opt_worker = _OptWorker(
            self.container, self.boxes,
            self._pop_spin.value(),
            self._gen_spin.value(),
            self._sa_spin.value(),
        )
        self._opt_worker.progress.connect(self._on_opt_progress)
        self._opt_worker.finished.connect(self._on_opt_finished)
        self._opt_worker.start()

    def _on_stop(self):
        if self._opt_worker:
            self._opt_worker.stop()

    def _on_reset(self):
        self.placed_boxes = []
        if OPENGL_OK:
            self.viewer.set_data(self.container, [])
            self.viewer.reset_view()
        self._populate_box_table()
        self._update_stats()
        self._progress.setValue(0)
        self._gen_label.setText("Phase: — / —")
        self._fit_label.setText("Best Fitness: —")

    # ── Box add / remove ────────────────────────────────────────

    def _on_add_box(self):
        dlg = AddBoxDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        name = dlg.name_edit.text().strip() or "New Box"
        new_id = max((b.id for b in self.boxes), default=0) + 1
        color = BOX_COLORS[(new_id - 1) % len(BOX_COLORS)]
        box = Box(
            id=new_id,
            name=name,
            width=float(dlg.width_spin.value()),
            depth=float(dlg.depth_spin.value()),
            height=float(dlg.height_spin.value()),
            weight=dlg.weight_spin.value(),
            can_rotate=dlg.rotate_check.isChecked(),
            color=color,
        )
        self.boxes.append(box)
        self._populate_box_table()
        self.statusBar().showMessage(f"Added box '{box.name}'")

    def _on_remove_box(self):
        row = self._table.currentRow()
        if row < 0 or row >= len(self.boxes):
            QMessageBox.information(self, "Remove Box",
                                    "Select a box in the list first.")
            return
        name = self.boxes[row].name
        del self.boxes[row]
        self._populate_box_table()
        self._update_stats()
        self.statusBar().showMessage(f"Removed box '{name}'")

    # ── 3-D box click handler ─────────────────────────────────────

    def _on_box_clicked(self, idx: int):
        """Open an info/edit dialog for the placed box at *idx*."""
        if idx < 0 or idx >= len(self.placed_boxes):
            return                    # clicked empty space – nothing to do

        pb = self.placed_boxes[idx]
        dlg = BoxInfoDialog(pb, self)
        if dlg.exec_() != QDialog.Accepted:
            return

        new_name  = dlg.get_name()
        new_color = dlg.get_color()

        # Update the *copy* kept in placed_boxes (used for drawing)
        pb.box.name  = new_name
        pb.box.color = new_color

        # Also update the *original* Box in self.boxes (matched by id)
        for box in self.boxes:
            if box.id == pb.box.id:
                box.name  = new_name
                box.color = new_color
                break

        # Refresh everything
        self._populate_box_table()
        if OPENGL_OK:
            self.viewer.set_data(self.container, self.placed_boxes)
            self.viewer.select_box(idx)
        self.statusBar().showMessage(
            f"Updated box '{new_name}'")

    # ── Optimiser progress / finished ────────────────────────────

    def _on_opt_progress(self, step, total, fitness, placed, phase):
        pct = int((step + 1) / max(total, 1) * 100)
        self._progress.setValue(min(pct, 99))
        self._gen_label.setText(phase)
        self._fit_label.setText(f"Best Fitness: {fitness:.4f}")

        if placed is not None:
            self.placed_boxes = placed
            if OPENGL_OK:
                self.viewer.set_data(self.container, placed)
            self._update_stats()
            self._populate_box_table()

    def _on_opt_finished(self, placed, fitness):
        self.placed_boxes = placed
        if OPENGL_OK:
            self.viewer.set_data(self.container, placed)
        self._populate_box_table()
        self._update_stats()

        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setValue(100)
        self._gen_label.setText("Completed  (GA + SA)")
        self._fit_label.setText(f"Best Fitness: {fitness:.4f}")
        self.statusBar().showMessage(
            f"Done -- placed {len(placed)}/{len(self.boxes)} boxes, "
            f"fitness {fitness:.4f}")
