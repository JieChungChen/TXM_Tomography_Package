from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt


class ContrastDialog(QDialog):
    def __init__(self, init_clip_lower=0.0, init_clip_upper=0.5, live_update_callback=None, parent=None):
        """
        Args:
            init_clip_lower: 初始下限裁切百分比 (0-10%)
            init_clip_upper: 初始上限裁切百分比 (0-10%)
            live_update_callback: 回呼函式 (clip_lower, clip_upper)
            parent: 父層元件
        """
        super().__init__(parent)
        self.setWindowTitle("Adjust Contrast")
        self.setFixedSize(400, 250)

        # 統一 Dialog 外觀
        self.setStyleSheet("""
            QDialog {
                border: 1px solid #e2e2e2;
                border-radius: 12px;
                background: #fafbfc;
            }
        """)
        from PyQt5.QtGui import QFont
        font = QFont("Calibri", 12)
        self.setFont(font)

        self.live_update_callback = live_update_callback

        # 統一滑桿樣式
        slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #bfbfbf;
                height: 6px;
                border-radius: 3px;
                background: #dedede;
            }
            QSlider::handle:horizontal {
                background: #1f6feb;
                border: none;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """

        # 下限裁切滑桿（最暗像素）。
        self.lower_slider = QSlider(Qt.Horizontal)
        self.lower_slider.setMinimum(0)
        self.lower_slider.setMaximum(100)  # 0.0% ~ 10.0%
        self.lower_slider.setValue(int(init_clip_lower * 10))
        self.lower_slider.valueChanged.connect(self.on_value_changed)
        self.lower_slider.setStyleSheet(slider_style)

        # 上限裁切滑桿（最亮像素）。
        self.upper_slider = QSlider(Qt.Horizontal)
        self.upper_slider.setMinimum(0)
        self.upper_slider.setMaximum(100)  # 0.0% ~ 10.0%
        self.upper_slider.setValue(int(init_clip_upper * 10))
        self.upper_slider.valueChanged.connect(self.on_value_changed)
        self.upper_slider.setStyleSheet(slider_style)

        # 標籤。
        self.lower_label = QLabel()
        self.upper_label = QLabel()
        self.update_labels()

        # 版面配置。
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # 下限裁切區塊。
        lower_title = QLabel("<b>Lower Bound (Clip Darkest Pixels)</b>")
        lower_title.setStyleSheet("font-family: Calibri; font-size: 14pt;")
        layout.addWidget(lower_title)
        layout.addWidget(self.lower_slider)
        self.lower_label.setStyleSheet("font-family: Calibri; font-size: 14pt; color: #333;")
        layout.addWidget(self.lower_label)

        layout.addSpacing(20)

        # 上限裁切區塊。
        upper_title = QLabel("<b>Upper Bound (Clip Brightest Pixels)</b>")
        upper_title.setStyleSheet("font-family: Calibri; font-size: 14pt;")
        layout.addWidget(upper_title)
        layout.addWidget(self.upper_slider)
        self.upper_label.setStyleSheet("font-family: Calibri; font-size: 14pt; color: #333;")
        layout.addWidget(self.upper_label)

        layout.addSpacing(15)
        self.setLayout(layout)

    def update_labels(self):
        """更新裁切百分比標籤。"""
        lower_val = self.lower_slider.value() / 10.0
        upper_val = self.upper_slider.value() / 10.0
        self.lower_label.setText(f"Clip bottom {lower_val:.1f}% darkest pixels")
        self.upper_label.setText(f"Clip top {upper_val:.1f}% brightest pixels")

    def on_value_changed(self):
        """處理滑桿數值變更。"""
        self.update_labels()

        if self.live_update_callback:
            clip_lower = self.lower_slider.value() / 10.0
            clip_upper = self.upper_slider.value() / 10.0
            self.live_update_callback(clip_lower, clip_upper)