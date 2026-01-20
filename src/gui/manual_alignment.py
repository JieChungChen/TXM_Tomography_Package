import numpy as np
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import (QLabel, QDialog, QPushButton, QVBoxLayout, QSizePolicy,
                             QHBoxLayout, QSlider, QFileDialog)
from PyQt5.QtCore import Qt, QEvent, QPoint
from PyQt5.QtGui import QImage, QPixmap
from src.logic.utils import norm_to_8bit


class AlignViewer(QDialog):
    def __init__(self, tomography, last_dir='.'):
        super().__init__()
        self.setWindowTitle("Align Viewer")
        self.setMinimumSize(500, 500)
        self.setModal(True)

        # 載入影像。
        self.tomo = tomography
        self.proj_images = [norm_to_8bit(img.copy()) for img in self.tomo.get_array()]
        self.last_dir = last_dir

        # 初始化斷層變數。
        self.index = len(self.proj_images) // 2
        self.shifts = [[0, 0] for _ in range(len(self.proj_images))]
        self.line_y1 = 100
        self.dragging_line1 = False

        # 初始化正弦圖變數。
        self.sino_array = None          # 目前顯示的正弦圖（RGB）
        self.sino_pixmap_size = (1, 1)  # (w, h) 實際貼到 QLabel 的大小
        self.sino_pixmap_offset = (0, 0)  # 圖片在 QLabel 內的偏移（置中）
        self.sino_crop = None           # (y0, y1, x0, x1) 目前裁切視窗
        self.sino_selecting = False
        self.sino_sel_start = None

        # 左側影像視窗。
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 右側即時正弦圖視窗。
        self.sino_label = QLabel("Sinogram")
        self.sino_label.setAlignment(Qt.AlignCenter)
        self.sino_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.prev_btn = QPushButton('Prev')
        self.next_btn = QPushButton('Next')
        self.save_btn = QPushButton('Save shifts')
        self.load_btn = QPushButton('Load shifts')
        self.done_btn = QPushButton('Finish') 
        self.reset_sino_btn = QPushButton('Reset view')

        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.save_btn.clicked.connect(self.save_shifts)
        self.load_btn.clicked.connect(self.load_shifts) 
        self.done_btn.clicked.connect(self.finish) 
        self.reset_sino_btn.clicked.connect(self.reset_sinogram_view)
        
        # 滑桿用於瀏覽影像。
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.proj_images) - 1)
        self.slider.setValue(self.index)
        self.slider.valueChanged.connect(self.slider_changed)

        # 版面配置。
        top_hbox = QHBoxLayout()
        top_hbox.addWidget(self.img_label, 3)
        top_hbox.addWidget(self.sino_label, 2)

        layout = QVBoxLayout()
        layout.addLayout(top_hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.prev_btn)
        hbox.addWidget(self.next_btn)
        hbox.addWidget(self.save_btn)
        hbox.addWidget(self.load_btn) 
        hbox.addWidget(self.done_btn)
        hbox.addWidget(self.reset_sino_btn) 
        layout.addLayout(hbox)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        self.sino_label.installEventFilter(self)

        self.update_view()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_view()

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

        self.update_view()

    def finish(self):
        for i, img in enumerate(self.proj_images):
            self.tomo.set(i, img)
        super().accept()
            
    def slider_changed(self, value):
        self.index = value
        self.update_view()

    def _clamp_sino_pos(self, pos: QPoint):
        """回傳夾到正弦圖區域邊界的 (x, y) 或 None（若無正弦圖）。"""
        if self.sino_array is None:
            return None
        x = pos.x() - self.sino_pixmap_offset[0]
        y = pos.y() - self.sino_pixmap_offset[1]
        pm_w, pm_h = self.sino_pixmap_size
        if pm_w <= 0 or pm_h <= 0:
            return None
        x = max(0, min(x, pm_w - 1))
        y = max(0, min(y, pm_h - 1))
        arr_h, arr_w = self.sino_array.shape[:2]
        arr_x = int(x * arr_w / pm_w)
        arr_y = int(y * arr_h / pm_h)
        arr_x = max(0, min(arr_x, arr_w - 1))
        arr_y = max(0, min(arr_y, arr_h - 1))
        return arr_x, arr_y

    def eventFilter(self, obj, event):
        if obj is self.sino_label:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                coord = self._clamp_sino_pos(event.pos())
                if coord is not None:
                    self.sino_selecting = True
                    self.sino_sel_start = coord
                    return True
            elif event.type() == QEvent.MouseMove and self.sino_selecting:
                coord = self._clamp_sino_pos(event.pos())
                if coord is not None and self.sino_sel_start is not None:
                    x0, y0 = self.sino_sel_start
                    x1, y1 = coord
                    self.update_sinogram(draw_rect=(min(x0, x1), min(y0, y1),
                                                    max(x0, x1), max(y0, y1)))
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self.sino_selecting and self.sino_sel_start is not None:
                    coord = self._clamp_sino_pos(event.pos())
                    if coord is not None:
                        x0, y0 = self.sino_sel_start
                        x1, y1 = coord
                        self.sino_crop = (min(y0, y1), max(y0, y1),
                                          min(x0, x1), max(x0, x1))
                    self.update_sinogram()
                self.sino_selecting = False
                self.sino_sel_start = None
                return True
        return super().eventFilter(obj, event)
    
    def reset_sinogram_view(self):
        self.sino_crop = None
        self.sino_selecting = False
        self.sino_sel_start = None
        self.update_sinogram()

    def keyPressEvent(self, event):
        key = event.key()
        dx, dy = 0, 0
        if key == Qt.Key_W:
            self.shifts[self.index][0] -= 1
            dy -= 1
        elif key == Qt.Key_S:
            self.shifts[self.index][0] += 1
            dy += 1
        elif key == Qt.Key_A:
            self.shifts[self.index][1] -= 1
            dx -= 1
        elif key == Qt.Key_D:
            self.shifts[self.index][1] += 1
            dx += 1

        img_temp = self.proj_images[self.index]
        self.proj_images[self.index] = np.roll(img_temp, shift=(dy, dx), axis=(0, 1))
        self.update_view()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.img_label.pixmap() is None:
            return
        
        # 滑鼠座標。
        mouse_x = event.x() - self.img_label.x()
        mouse_y = event.y() - self.img_label.y()
        pixmap = self.img_label.pixmap()
        label_w, label_h = self.img_label.width(), self.img_label.height()
        pm_w, pm_h = pixmap.width(), pixmap.height()
        # 圖片在 QLabel 中的偏移（置中）。
        offset_x = max((label_w - pm_w) // 2, 0)
        offset_y = max((label_h - pm_h) // 2, 0)
        # 扣除 padding 得到實際圖片上的座標。
        img_x = int((mouse_x - offset_x) / self.scale)
        img_y = int((mouse_y - offset_y) / self.scale)

        # 防止座標超出範圍。
        if img_x < 0 or img_x >= self.raw_width or img_y < 0 or img_y >= self.raw_height:
            return

        if abs(img_y - self.line_y1) < 5:
            self.dragging_line1 = True
        else:
            center_x = self.raw_width // 2
            center_y = self.raw_height // 2
            dx = center_x - img_x
            dy = center_y - img_y

            self.shifts[self.index][0] += dy
            self.shifts[self.index][1] += dx
            self.proj_images[self.index] = np.roll(self.proj_images[self.index], shift=(dy, dx), axis=(0, 1))
            self.update_view()

    def mouseMoveEvent(self, event):
        if self.dragging_line1:
            mouse_y = event.pos().y() - self.img_label.pos().y()
            img_y = int(mouse_y / self.scale)
            img_y = max(0, min(img_y, self.raw_height - 1))
            self.line_y1 = img_y
            self.update_view()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging_line1 = False

    def update_view(self):
        img_show = self.proj_images[self.index]
        img_rgb = np.stack([img_show]*3, axis=-1)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        h, w = pil_img.size
        scale = 512 // max(h, w)
        center_x, center_y = w // 2, h // 2
        
        # 輔助線。
        draw.line([(0, self.line_y1), (pil_img.width, self.line_y1)], fill=(0, 255, 0), width=1)
        # 中心十字輔助線。
        r = 8 / scale
        draw.line([(center_x - r, center_y), (center_x + r, center_y)], fill=(0, 255, 0), width=2//scale)
        draw.line([(center_x, center_y - r), (center_x, center_y + r)], fill=(0, 255, 0), width=2//scale)

        self.raw_height, self.raw_width = h, w
        label_w, label_h = self.img_label.width(), self.img_label.height()
        scale_w = label_w / self.raw_width
        scale_h = label_h / self.raw_height
        self.scale = min(scale_w, scale_h)
        
        img_rgb = np.array(pil_img)
        qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.img_label.setPixmap(pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.update_sinogram()
        self.setWindowTitle(f"Align Viewer {self.index + 1}/{len(self.proj_images)}")

    def update_sinogram(self, row_index=None, draw_rect=None):
        if row_index is None:
            row_index = int(self.line_y1)
        row_index = max(0, min(row_index, self.raw_height - 1))

        try:
            sino_full = np.stack([img[row_index, :] for img in self.proj_images], axis=1)  # (width, N_proj)
        except Exception:
            return

        # 裁切視窗（若有選擇）。
        y0, y1, x0, x1 = 0, sino_full.shape[0], 0, sino_full.shape[1]
        if self.sino_crop:
            y0, y1, x0, x1 = self.sino_crop
            # 邊界修正。
            y0 = max(0, min(y0, sino_full.shape[0]-1))
            y1 = max(y0+1, min(y1, sino_full.shape[0]))
            x0 = max(0, min(x0, sino_full.shape[1]-1))
            x1 = max(x0+1, min(x1, sino_full.shape[1]))
        sino = sino_full[y0:y1, x0:x1]

        sino8 = norm_to_8bit(sino)
        rgb = np.repeat(sino8[:, :, None], 3, axis=2)

        # 單一向下箭頭，指示目前投影。
        arrow_x = self.index - x0
        if 0 <= arrow_x < rgb.shape[1]:
            tip_h = max(2, min(rgb.shape[0], 8))
            for dy in range(tip_h):
                span = tip_h - dy
                lx = arrow_x - span + 1
                rx = arrow_x + span
                lx = max(lx, 0)
                rx = min(rx, rgb.shape[1])
                rgb[dy, lx:rx, :] = [255, 0, 0]

        # 若正在框選，畫出矩形預覽。
        if draw_rect is not None:
            rx0, ry0, rx1, ry1 = draw_rect
            rx0 = max(0, min(rx0, rgb.shape[1]-1))
            rx1 = max(0, min(rx1, rgb.shape[1]-1))
            ry0 = max(0, min(ry0, rgb.shape[0]-1))
            ry1 = max(0, min(ry1, rgb.shape[0]-1))
            rgb[ry0:ry1+1, [rx0, rx1], :] = [0, 255, 0]
            rgb[[ry0, ry1], rx0:rx1+1, :] = [0, 255, 0]

        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # 紀錄尺寸與偏移，用於座標換算。
        label_w, label_h = self.sino_label.width(), self.sino_label.height()
        pm = pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.sino_pixmap_size = (pm.width(), pm.height())
        self.sino_pixmap_offset = (max((label_w - pm.width()) // 2, 0),
                                   max((label_h - pm.height()) // 2, 0))
        self.sino_array = rgb
        self.sino_label.setPixmap(pm)
    