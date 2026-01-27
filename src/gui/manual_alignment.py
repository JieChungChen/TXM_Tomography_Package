import numpy as np
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import (QLabel, QDialog, QPushButton, QVBoxLayout, QSizePolicy,
                             QHBoxLayout, QSlider, QFileDialog)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont
from src.logic.utils import norm_to_8bit


class AlignViewer(QDialog):
    FONT_ZOOM = QFont('Calibri', 14, QFont.Bold)
    FONT_CTRL = QFont('Calibri', 14)
    SLIDER_STYLE = """
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

    def __init__(self, tomography, last_dir='.'): 
        super().__init__()
        self.setWindowTitle("Align Viewer")
        self.setFixedSize(1600, 900)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                border: 1px solid #e2e2e2;
                border-radius: 12px;
                background: #fafbfc;
            }
        """)

        # load tomography data
        self.tomo = tomography
        self.proj_images = [norm_to_8bit(img.copy()) for img in self.tomo.get_full_images()]
        self.last_dir = last_dir
        self.n_proj = len(self.proj_images)
        self.raw_size = self.proj_images[0].shape[0]
        self.rotational_center = {'x': self.raw_size // 2, 'y': self.raw_size // 2}

        # GUI variables
        self.view_scale = 1.0
        self.max_scale = 3.0
        self.min_scale = 0.75
        self.index = len(self.proj_images) // 2
        self.shifts = [[0, 0] for _ in range(len(self.proj_images))]
        self.line_y = 100
        self.dragging_line = False

        # sinogram variables
        self.sino_array = None
        self.sino_pixmap_size = (1, 1)
        self.sino_pixmap_offset = (0, 0)
        self.sino_crop = None
        self.sino_selecting = False
        self.sino_sel_start = None

        # initialize GUI
        self._init_labels()
        self._init_buttons()
        self._init_slider()
        self._setup_layout()
        self.sino_label.installEventFilter(self)
        self.update_all()

    def _init_labels(self):
        self.img_label = QLabel("Tomography")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.hs_label = QLabel("Horizontal Sum")
        self.hs_label.setAlignment(Qt.AlignCenter)
        self.hs_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.sino_label = QLabel("Sinogram")
        self.sino_label.setAlignment(Qt.AlignCenter)
        self.sino_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _init_buttons(self):
        # zoom in / out bottons
        self.zoom_in_btn = QPushButton('Zoom In')
        self.zoom_in_btn.setFont(self.FONT_ZOOM)
        self.zoom_in_btn.setFixedSize(100, 40)
        self.zoom_in_btn.clicked.connect(self.zoom_in)

        self.zoom_out_btn = QPushButton('Zoom Out')
        self.zoom_out_btn.setFont(self.FONT_ZOOM)
        self.zoom_out_btn.setFixedSize(100, 40)
        self.zoom_out_btn.clicked.connect(self.zoom_out)

        # control buttons
        self.prev_btn = QPushButton('Prev')
        self.next_btn = QPushButton('Next')
        self.save_btn = QPushButton('Save shifts')
        self.save_btn.setToolTip('Save current shifts to a text file')
        self.load_btn = QPushButton('Load shifts')
        self.load_btn.setToolTip('Load shifts from a text file')
        self.done_btn = QPushButton('Finish')
        self.reset_sino_btn = QPushButton('Reset sino')
        self.reset_sino_btn.setToolTip('Reset sinogram view')

        for btn in [self.prev_btn, self.next_btn, self.save_btn, 
                    self.load_btn, self.done_btn, self.reset_sino_btn]:
            btn.setFont(self.FONT_CTRL)

        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.save_btn.clicked.connect(self.save_shifts)
        self.load_btn.clicked.connect(self.load_shifts)
        self.done_btn.clicked.connect(self.finish)
        self.reset_sino_btn.clicked.connect(self.reset_sino_view)

    def _init_slider(self):
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet(self.SLIDER_STYLE)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.proj_images) - 1)
        self.slider.setValue(self.index)
        self.slider.valueChanged.connect(self.slider_changed)

    def _setup_layout(self):
        # zoom buttons
        zoom_hbox = QHBoxLayout()
        zoom_hbox.addStretch(1)
        zoom_hbox.addWidget(self.reset_sino_btn)
        zoom_hbox.addWidget(self.zoom_in_btn)
        zoom_hbox.addWidget(self.zoom_out_btn)

        # images (tomo & sino) with reset button under sino
        tomo_vbox = QVBoxLayout()
        tomo_vbox.addWidget(self.img_label, 3)

        hs_vbox = QVBoxLayout()
        hs_vbox.addWidget(self.hs_label)

        sino_vbox = QVBoxLayout()
        sino_vbox.addWidget(self.sino_label)

        top_hbox = QHBoxLayout()
        top_hbox.addLayout(tomo_vbox, 3)
        top_hbox.addLayout(hs_vbox, 2)
        top_hbox.addLayout(sino_vbox, 2)

        layout = QVBoxLayout()
        layout.addLayout(zoom_hbox)
        layout.addLayout(top_hbox)
        # control buttons (without reset)
        hbox = QHBoxLayout()
        hbox.addWidget(self.prev_btn)
        hbox.addWidget(self.next_btn)
        hbox.addWidget(self.save_btn)
        hbox.addWidget(self.load_btn)
        hbox.addWidget(self.done_btn)
        layout.addLayout(hbox)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def zoom_in(self):
        if self.view_scale < self.max_scale:
            self.view_scale *= 1.25
            self.resize_view()

    def zoom_out(self):
        if self.view_scale > self.min_scale:
            self.view_scale /= 1.25
            self.resize_view()

    def resize_view(self):
        base_w, base_h = 1600, 900
        new_w = int(base_w * self.view_scale)
        new_h = int(base_h * self.view_scale)
        self.setFixedSize(new_w, new_h)
        self.update_all()

    def slider_changed(self, value):
        self.index = value
        self.update_tomo()
        self.update_sino()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.slider.setValue(self.index)

    def next_image(self):
        if self.index < len(self.proj_images) - 1:
            self.index += 1
            self.slider.setValue(self.index)

    def save_shifts(self):
        save_path = QFileDialog.getSaveFileName(self, "Save shifts", self.last_dir, "Text Files (*.txt)")[0]
        if save_path:
            with open(save_path, 'w') as f:
                for i, (dx, dy) in enumerate(self.shifts):
                    f.write(f"{str(i).zfill(3)},{dx},{dy}\n")

    def load_shifts(self):
        shifts_file, _ = QFileDialog.getOpenFileName(None, "Load Shifts", "*.txt")
        if shifts_file:
            with open(shifts_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    try:
                        idx, dy, dx = map(int, parts)
                        if 0 <= idx < len(self.shifts):
                            self.shifts[idx] = [dy, dx]  
                            self.proj_images[idx] = np.roll(self.proj_images[idx], shift=(dy, dx), axis=(0, 1))
                    except Exception:
                        continue
        self.update_all()

    def finish(self):
        for i, img in enumerate(self.proj_images):
            self.tomo.set(i, img)
        super().accept()

    def reset_sino_view(self):
        self.sino_crop = None
        self.sino_selecting = False
        self.sino_sel_start = None
        self.update_sino()

    def update_all(self):
        self.update_tomo()
        self.update_sino()
        self.update_horizontal_sum()

    # -------------- core logic --------------- 
    def get_sino_pos(self, pos):
        """
        get the actual sino position when clicking on the sino image.
        """
        x = pos.x() - self.sino_pixmap_offset[0]
        y = pos.y() - self.sino_pixmap_offset[1]
        pm_w, pm_h = self.sino_pixmap_size
        x = max(0, min(x, pm_w))
        y = max(0, min(y, pm_h))
        arr_h, arr_w = self.sino_array.shape[:2]
        arr_x = int(x * arr_w / pm_w)
        arr_y = int(y * arr_h / pm_h)
        arr_x = max(0, min(arr_x, arr_w))
        arr_y = max(0, min(arr_y, arr_h))
        return arr_x, arr_y

    def eventFilter(self, obj, event):
        if obj is self.sino_label:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                coord = self.get_sino_pos(event.pos())
                if coord is not None:
                    self.sino_selecting = True
                    self.sino_sel_start = coord
                    return True
            elif event.type() == QEvent.MouseMove and self.sino_selecting:
                coord = self.get_sino_pos(event.pos())
                if coord is not None and self.sino_sel_start is not None:
                    x0, y0 = self.sino_sel_start
                    x1, y1 = coord
                    self.update_sino(draw_rect=(min(x0, x1), min(y0, y1),
                                                max(x0, x1), max(y0, y1)))
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self.sino_selecting and self.sino_sel_start is not None:
                    coord = self.get_sino_pos(event.pos())
                    if coord is not None:
                        x0, y0 = self.sino_sel_start
                        x1, y1 = coord
                        # Only set crop if dragged (start != end)
                        if (x0, y0) != (x1, y1):
                            self.sino_crop = (min(y0, y1), max(y0, y1),
                                              min(x0, x1), max(x0, x1))
                    self.update_sino()
                self.sino_selecting = False
                self.sino_sel_start = None
                return True
        return super().eventFilter(obj, event)
    
    def keyPressEvent(self, event):
        key_map = {
            Qt.Key_W: (0, -1),
            Qt.Key_S: (0, 1),
            Qt.Key_A: (1, -1),
            Qt.Key_D: (1, 1),
        }
        key = event.key()
        if key in key_map:
            axis, delta = key_map[key]
            self.shifts[self.index][axis] += delta
            # 依照 axis 決定 dx, dy
            dy = self.shifts[self.index][0] - self.shifts[self.index][0] + (delta if axis == 0 else 0)
            dx = self.shifts[self.index][1] - self.shifts[self.index][1] + (delta if axis == 1 else 0)
            # 直接用 axis 判斷
            dy = delta if axis == 0 else 0
            dx = delta if axis == 1 else 0
            img_temp = self.proj_images[self.index]
            self.proj_images[self.index] = np.roll(img_temp, shift=(dy, dx), axis=(0, 1))
            self.update_all()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.img_label.pixmap() is None:
            return
        
        # mouse coordinates
        mouse_x = event.x() - self.img_label.x()
        mouse_y = event.y() - self.img_label.y()
        pixmap = self.img_label.pixmap()
        label_w, label_h = self.img_label.width(), self.img_label.height()
        pm_w, pm_h = pixmap.width(), pixmap.height()
        offset_x = max((label_w - pm_w) // 2, 0)
        offset_y = max((label_h - pm_h) // 2, 0)
        img_x = int((mouse_x - offset_x) / self.scale)
        img_y = int((mouse_y - offset_y) / self.scale)

        # prevent out-of-bounds
        if img_x < 0 or img_x >= self.raw_size or img_y < 0 or img_y >= self.raw_size:
            return

        # click on green line
        if abs(img_y - self.line_y) < 5:
            self.dragging_line = True
        # click on tomo image
        else:
            dx = self.rotational_center['x'] - img_x
            dy = self.rotational_center['y'] - img_y
            self.shifts[self.index][0] += dy
            self.shifts[self.index][1] += dx
            self.proj_images[self.index] = np.roll(self.proj_images[self.index], shift=(dy, dx), axis=(0, 1))
            self.update_all()

    def mouseMoveEvent(self, event):
        if self.dragging_line:
            mouse_y = event.pos().y() - self.img_label.pos().y()
            img_y = int(mouse_y / self.scale)
            img_y = max(0, min(img_y, self.raw_size - 1))
            self.line_y = img_y
            self.update_tomo()
            self.update_sino()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging_line = False

    def update_tomo(self):
        size = self.raw_size
        center_x, center_y = self.rotational_center['x'], self.rotational_center['y']
        img_rgb = np.stack([self.proj_images[self.index]]*3, axis=-1)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # support green line
        line_scale = 512 // size
        draw.line([(0, self.line_y), (pil_img.width, self.line_y)], fill=(0, 255, 0), width=1)
        # crosshair at rotational center
        r = 8 / line_scale
        draw.line([(center_x - r, center_y), (center_x + r, center_y)], fill=(0, 255, 0), width=2//line_scale)
        draw.line([(center_x, center_y - r), (center_x, center_y + r)], fill=(0, 255, 0), width=2//line_scale)
        img_rgb = np.array(pil_img)

        # distribute space of three labels
        total_w = self.width() - 40  # 預留邊界
        total_h = self.height() - 180  # 預留上方/下方空間
        img_label_w = int(total_w * 0.5)
        hs_label_w = int(total_w * 0.2) 
        sino_label_w = total_w - img_label_w - hs_label_w
        img_label_h = hs_label_h = sino_label_h = total_h
        self.img_label.setFixedSize(img_label_w, img_label_h)
        self.hs_label.setFixedSize(hs_label_w, hs_label_h)
        self.sino_label.setFixedSize(sino_label_w, sino_label_h)
        scale_w = img_label_w / size
        scale_h = img_label_h / size
        self.scale = min(scale_w, scale_h)

        qimg = QImage(img_rgb.data, size, size, 3 * size, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.img_label.setPixmap(pixmap.scaled(img_label_w, img_label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.update_sino()
        self.setWindowTitle(f"Align Viewer {self.index + 1}/{len(self.proj_images)}")

    def update_horizontal_sum(self):
        tomo = np.array(self.proj_images)
        hs = tomo.sum(axis=2).T  # shape: (H, n_projs)
        hs_8bit = norm_to_8bit(hs)
        hs_8bit = Image.fromarray(hs_8bit).resize((self.n_proj, 512), Image.Resampling.LANCZOS)
        hs_rgb = np.repeat(np.array(hs_8bit)[:, :, None], 3, axis=2)
        h, w = hs_rgb.shape[:2]
        qimg = QImage(hs_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        label_w, label_h = self.hs_label.width(), self.hs_label.height()
        pixmap = QPixmap.fromImage(qimg).scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.hs_label.setPixmap(pixmap)

    def update_sino(self, draw_rect=None):
        """
        Parameters
        ----------
        draw_rect : tuple or None
            If provided, should be (x0, y0, x1, y1) to draw a rectangle on the sinogram preview.
        """
        row_index = int(self.line_y)
        sino_full = np.stack([img[row_index, :] for img in self.proj_images], axis=1)  # (width, N_proj)

        # crop window
        y0, y1, x0, x1 = 0, sino_full.shape[0], 0, sino_full.shape[1]
        if self.sino_crop:
            y0, y1, x0, x1 = self.sino_crop
            y0 = max(0, min(y0, sino_full.shape[0]-1))
            y1 = max(y0+1, min(y1, sino_full.shape[0]))
            x0 = max(0, min(x0, sino_full.shape[1]-1))
            x1 = max(x0+1, min(x1, sino_full.shape[1]))
        sino = sino_full[y0:y1, x0:x1]

        sino_8bit = norm_to_8bit(sino)
        sino_rgb = np.repeat(sino_8bit[:, :, None], 3, axis=2)

        # 單一向下箭頭，指示目前投影。
        arrow_x = self.index - x0
        if 0 <= arrow_x < sino_rgb.shape[1]:
            tip_h = max(2, min(sino_rgb.shape[0], 8))
            for dy in range(tip_h):
                span = tip_h - dy
                lx = arrow_x - span + 1
                rx = arrow_x + span
                lx = max(lx, 0)
                rx = min(rx, sino_rgb.shape[1])
                sino_rgb[dy, lx:rx, :] = [255, 0, 0]

        # 若正在框選，畫出矩形預覽。
        if draw_rect is not None:
            rx0, ry0, rx1, ry1 = draw_rect
            rx0 = max(0, min(rx0, sino_rgb.shape[1]-1))
            rx1 = max(0, min(rx1, sino_rgb.shape[1]-1))
            ry0 = max(0, min(ry0, sino_rgb.shape[0]-1))
            ry1 = max(0, min(ry1, sino_rgb.shape[0]-1))
            sino_rgb[ry0:ry1+1, [rx0, rx1], :] = [0, 255, 0]
            sino_rgb[[ry0, ry1], rx0:rx1+1, :] = [0, 255, 0]

        h, w = sino_rgb.shape[:2]
        qimg = QImage(sino_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # 紀錄尺寸與偏移，用於座標換算。
        label_w, label_h = self.sino_label.width(), self.sino_label.height()
        pm = pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.sino_pixmap_size = (pm.width(), pm.height())
        self.sino_pixmap_offset = (max((label_w - pm.width()) // 2, 0),
                                   max((label_h - pm.height()) // 2, 0))
        self.sino_array = sino_rgb
        self.sino_label.setPixmap(pm)
    