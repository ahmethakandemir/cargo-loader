"""OpenGL 3-D viewer widget for the container and placed boxes.

Controls
--------
* **Left-click**  – select a box (opens info dialog via signal)
* **Left-drag**   – rotate the camera
* **Right-drag**  – pan
* **Scroll wheel** – zoom in / out
"""

import math
from typing import List, Optional

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QMouseEvent, QWheelEvent

try:
    from OpenGL.GL import *   # noqa: F401,F403
    from OpenGL.GLU import *  # noqa: F401,F403
    OPENGL_OK = True
except ImportError:
    OPENGL_OK = False

from .models import Container, PlacedBox


# ── Ray-AABB intersection (slab method) ─────────────────────────────

def _ray_aabb(ox, oy, oz, dx, dy, dz, x0, y0, z0, x1, y1, z1):
    """Return the ray parameter *t* at which the ray first hits the AABB,
    or *None* if it misses.  The ray is  origin + t * direction."""
    tmin = -1e30
    tmax = 1e30
    for o, d, lo, hi in ((ox, dx, x0, x1),
                          (oy, dy, y0, y1),
                          (oz, dz, z0, z1)):
        if abs(d) < 1e-10:
            if o < lo or o > hi:
                return None
        else:
            t1 = (lo - o) / d
            t2 = (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    if tmax < 0:
        return None
    return tmin if tmin >= 0 else tmax


class ContainerViewer(QOpenGLWidget):
    """Interactive 3-D view of the container and packed boxes."""

    # Emitted when the user clicks on a box (index) or empty space (-1)
    box_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.container: Container = Container()
        self.placed_boxes: List[PlacedBox] = []

        # Camera state
        self._rot_x = 25.0
        self._rot_y = -35.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._zoom  = 1.0
        self._last_pos = QPoint()
        self._press_pos = QPoint()

        # Selection
        self._selected_idx: int = -1

        # Cached matrices for ray picking (set every paintGL)
        self._mv = None
        self._proj = None
        self._vp = None

        self.setMinimumSize(480, 360)
        self.setFocusPolicy(Qt.StrongFocus)

    # ── public API ────────────────────────────────────────────────

    def set_data(self, container: Container,
                 placed_boxes: Optional[List[PlacedBox]] = None) -> None:
        self.container = container
        self.placed_boxes = placed_boxes or []
        self.update()

    def reset_view(self) -> None:
        self._rot_x, self._rot_y = 25.0, -35.0
        self._pan_x = self._pan_y = 0.0
        self._zoom = 1.0
        self._selected_idx = -1
        self.update()

    def select_box(self, idx: int) -> None:
        """Programmatically select a placed box by index (-1 to clear)."""
        self._selected_idx = idx
        self.update()

    # ── OpenGL callbacks ──────────────────────────────────────────

    def initializeGL(self) -> None:
        if not OPENGL_OK:
            return
        glClearColor(0.11, 0.11, 0.16, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        # Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glLightfv(GL_LIGHT0, GL_POSITION, [0.5, 1.0, 0.8, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.22, 0.22, 0.25, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.80, 0.80, 0.76, 1.0])

        glLightfv(GL_LIGHT1, GL_POSITION, [-0.3, -0.5, -0.5, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.22, 0.22, 0.28, 1.0])

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def resizeGL(self, w: int, h: int) -> None:
        if not OPENGL_OK:
            return
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(h, 1), 1.0, 15000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self) -> None:
        if not OPENGL_OK:
            return
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        c = self.container
        diag = math.sqrt(c.width ** 2 + c.depth ** 2 + c.height ** 2)
        dist = diag * 1.3 * self._zoom

        glTranslatef(self._pan_x, self._pan_y, -dist)
        glRotatef(self._rot_x, 1.0, 0.0, 0.0)
        glRotatef(self._rot_y, 0.0, 1.0, 0.0)

        # Centre the container at the origin
        glTranslatef(-c.width / 2.0, -c.height / 2.0, -c.depth / 2.0)

        # Cache matrices for ray-based click picking
        self._mv   = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._proj = glGetDoublev(GL_PROJECTION_MATRIX)
        self._vp   = glGetIntegerv(GL_VIEWPORT)

        self._draw_floor_grid()
        self._draw_container_wireframe()
        self._draw_placed_boxes()
        self._draw_selection_highlight()

    # ── drawing helpers ───────────────────────────────────────────

    def _draw_floor_grid(self) -> None:
        c = self.container
        glDisable(GL_LIGHTING)
        glColor4f(0.28, 0.28, 0.32, 0.45)
        glLineWidth(1.0)
        step = 50.0
        glBegin(GL_LINES)
        x = 0.0
        while x <= c.width + 0.01:
            glVertex3f(x, 0.0, 0.0)
            glVertex3f(x, 0.0, c.depth)
            x += step
        z = 0.0
        while z <= c.depth + 0.01:
            glVertex3f(0.0, 0.0, z)
            glVertex3f(c.width, 0.0, z)
            z += step
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_container_wireframe(self) -> None:
        c = self.container
        w, h, d = c.width, c.height, c.depth
        glDisable(GL_LIGHTING)
        glColor4f(0.55, 0.65, 0.78, 0.85)
        glLineWidth(2.0)

        edges = [
            # bottom ring
            (0, 0, 0, w, 0, 0), (w, 0, 0, w, 0, d),
            (w, 0, d, 0, 0, d), (0, 0, d, 0, 0, 0),
            # top ring
            (0, h, 0, w, h, 0), (w, h, 0, w, h, d),
            (w, h, d, 0, h, d), (0, h, d, 0, h, 0),
            # verticals
            (0, 0, 0, 0, h, 0), (w, 0, 0, w, h, 0),
            (w, 0, d, w, h, d), (0, 0, d, 0, h, d),
        ]
        glBegin(GL_LINES)
        for e in edges:
            glVertex3f(e[0], e[1], e[2])
            glVertex3f(e[3], e[4], e[5])
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_placed_boxes(self) -> None:
        for pb in self.placed_boxes:
            r, g, b = pb.box.color
            self._draw_solid_box(pb.x, pb.y, pb.z,
                                 pb.width, pb.depth, pb.height,
                                 r, g, b)
            self._draw_box_edges(pb.x, pb.y, pb.z,
                                 pb.width, pb.depth, pb.height)

    @staticmethod
    def _draw_solid_box(x, y, z, w, d, h, r, g, b, alpha=0.88):
        glColor4f(r, g, b, alpha)
        glBegin(GL_QUADS)
        # Front  (z + d)
        glNormal3f(0, 0, 1)
        glVertex3f(x,     y,     z + d)
        glVertex3f(x + w, y,     z + d)
        glVertex3f(x + w, y + h, z + d)
        glVertex3f(x,     y + h, z + d)
        # Back   (z)
        glNormal3f(0, 0, -1)
        glVertex3f(x + w, y,     z)
        glVertex3f(x,     y,     z)
        glVertex3f(x,     y + h, z)
        glVertex3f(x + w, y + h, z)
        # Top    (y + h)
        glNormal3f(0, 1, 0)
        glVertex3f(x,     y + h, z)
        glVertex3f(x,     y + h, z + d)
        glVertex3f(x + w, y + h, z + d)
        glVertex3f(x + w, y + h, z)
        # Bottom (y)
        glNormal3f(0, -1, 0)
        glVertex3f(x,     y, z + d)
        glVertex3f(x,     y, z)
        glVertex3f(x + w, y, z)
        glVertex3f(x + w, y, z + d)
        # Right  (x + w)
        glNormal3f(1, 0, 0)
        glVertex3f(x + w, y,     z)
        glVertex3f(x + w, y,     z + d)
        glVertex3f(x + w, y + h, z + d)
        glVertex3f(x + w, y + h, z)
        # Left   (x)
        glNormal3f(-1, 0, 0)
        glVertex3f(x, y,     z + d)
        glVertex3f(x, y,     z)
        glVertex3f(x, y + h, z)
        glVertex3f(x, y + h, z + d)
        glEnd()

    @staticmethod
    def _draw_box_edges(x, y, z, w, d, h):
        glDisable(GL_LIGHTING)
        glColor4f(0.0, 0.0, 0.0, 0.55)
        glLineWidth(1.4)
        edges = [
            (x, y, z, x+w, y, z),         (x, y, z+d, x+w, y, z+d),
            (x, y+h, z, x+w, y+h, z),     (x, y+h, z+d, x+w, y+h, z+d),
            (x, y, z, x, y, z+d),         (x+w, y, z, x+w, y, z+d),
            (x, y+h, z, x, y+h, z+d),     (x+w, y+h, z, x+w, y+h, z+d),
            (x, y, z, x, y+h, z),         (x+w, y, z, x+w, y+h, z),
            (x, y, z+d, x, y+h, z+d),     (x+w, y, z+d, x+w, y+h, z+d),
        ]
        glBegin(GL_LINES)
        for e in edges:
            glVertex3f(e[0], e[1], e[2])
            glVertex3f(e[3], e[4], e[5])
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_selection_highlight(self) -> None:
        """Draw a bright outline around the currently selected box."""
        if self._selected_idx < 0 or self._selected_idx >= len(self.placed_boxes):
            return
        pb = self.placed_boxes[self._selected_idx]
        x, y, z = pb.x, pb.y, pb.z
        w, d, h = pb.width, pb.depth, pb.height

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)          # always on top
        glColor4f(1.0, 1.0, 0.15, 0.95)  # bright yellow
        glLineWidth(3.5)

        edges = [
            (x, y, z, x+w, y, z),         (x, y, z+d, x+w, y, z+d),
            (x, y+h, z, x+w, y+h, z),     (x, y+h, z+d, x+w, y+h, z+d),
            (x, y, z, x, y, z+d),         (x+w, y, z, x+w, y, z+d),
            (x, y+h, z, x, y+h, z+d),     (x+w, y+h, z, x+w, y+h, z+d),
            (x, y, z, x, y+h, z),         (x+w, y, z, x+w, y+h, z),
            (x, y, z+d, x, y+h, z+d),     (x+w, y, z+d, x+w, y+h, z+d),
        ]
        glBegin(GL_LINES)
        for e in edges:
            glVertex3f(e[0], e[1], e[2])
            glVertex3f(e[3], e[4], e[5])
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glLineWidth(1.0)

    # ── ray picking ──────────────────────────────────────────────

    def _pick_box(self, mx: int, my: int) -> int:
        """Cast a ray through screen position (*mx*, *my*) and return the
        index of the nearest placed box hit, or -1."""
        if self._mv is None or not self.placed_boxes:
            return -1
        try:
            vp = self._vp
            gl_y = float(int(vp[3]) - my - 1)
            near = gluUnProject(float(mx), gl_y, 0.0,
                                self._mv, self._proj, vp)
            far  = gluUnProject(float(mx), gl_y, 1.0,
                                self._mv, self._proj, vp)
        except Exception:
            return -1

        dx = far[0] - near[0]
        dy = far[1] - near[1]
        dz = far[2] - near[2]
        ln = math.sqrt(dx * dx + dy * dy + dz * dz)
        if ln < 1e-10:
            return -1
        dx /= ln;  dy /= ln;  dz /= ln

        best_t = float('inf')
        best_idx = -1
        for i, pb in enumerate(self.placed_boxes):
            t = _ray_aabb(near[0], near[1], near[2], dx, dy, dz,
                          pb.x, pb.y, pb.z,
                          pb.x + pb.width,
                          pb.y + pb.height,
                          pb.z + pb.depth)
            if t is not None and 0 <= t < best_t:
                best_t = t
                best_idx = i
        return best_idx

    # ── mouse interaction ─────────────────────────────────────────

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        self._last_pos = ev.pos()
        self._press_pos = ev.pos()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.LeftButton:
            dx = ev.x() - self._press_pos.x()
            dy = ev.y() - self._press_pos.y()
            if dx * dx + dy * dy < 25:        # click, not drag
                idx = self._pick_box(ev.x(), ev.y())
                self._selected_idx = idx
                self.update()
                self.box_clicked.emit(idx)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        dx = ev.x() - self._last_pos.x()
        dy = ev.y() - self._last_pos.y()

        if ev.buttons() & Qt.LeftButton:
            self._rot_x += dy * 0.5
            self._rot_y += dx * 0.5
            self.update()
        elif ev.buttons() & Qt.RightButton:
            self._pan_x += dx * 0.5
            self._pan_y -= dy * 0.5
            self.update()

        self._last_pos = ev.pos()

    def wheelEvent(self, ev: QWheelEvent) -> None:
        delta = ev.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._zoom = max(0.15, min(5.0, self._zoom * factor))
        self.update()
