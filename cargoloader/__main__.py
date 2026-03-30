"""Entry-point: ``python -m cargoloader``."""

import sys
import os

# ── Windows-specific tweaks ──────────────────────────────────────
if sys.platform == "win32":
    # Enable automatic high-DPI scaling on Windows 10+
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from PyQt5.QtWidgets import QApplication          # noqa: E402
from PyQt5.QtCore import Qt                        # noqa: E402
from PyQt5.QtGui import QPalette, QColor           # noqa: E402

from .mainwindow import MainWindow                 # noqa: E402


def _apply_dark_palette(app: QApplication) -> None:
    """Set a modern dark colour scheme via the Fusion style."""
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window,          QColor(45, 45, 48))
    p.setColor(QPalette.WindowText,      QColor(210, 210, 210))
    p.setColor(QPalette.Base,            QColor(30, 30, 33))
    p.setColor(QPalette.AlternateBase,   QColor(45, 45, 48))
    p.setColor(QPalette.ToolTipBase,     QColor(25, 25, 25))
    p.setColor(QPalette.ToolTipText,     QColor(210, 210, 210))
    p.setColor(QPalette.Text,            QColor(210, 210, 210))
    p.setColor(QPalette.Button,          QColor(45, 45, 48))
    p.setColor(QPalette.ButtonText,      QColor(210, 210, 210))
    p.setColor(QPalette.BrightText,      QColor(255, 60, 60))
    p.setColor(QPalette.Link,            QColor(42, 130, 218))
    p.setColor(QPalette.Highlight,       QColor(42, 130, 218))
    p.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(p)


def main() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    _apply_dark_palette(app)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
