import numpy as np
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import (QLabel, QDialog, QPushButton, QVBoxLayout, QSizePolicy,
                             QHBoxLayout, QSlider, QFileDialog)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor
from src.gui.cc_align_dialog import CCAlignDialog 
from src.logic.utils import norm_hs_to_8bit


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

        # load tomography data
        self.tomo = tomography
        self.proj_images = self.tomo.get_norm_images()
        self.last_dir = last_dir
        self.n_proj, self.raw_size, _ = self.proj_images.shape
        self.rotational_center = (self.raw_size // 2, self.raw_size // 2)

        # GUI variables
        self.view_scale = 1.0
        self.max_scale = 3.0
        self.min_scale = 0.75
        self.index = self.n_proj // 2
        self.line_color = (0, 255, 0, 160)
        self.line_y = 100
        self.changing_center = False
        self.tomo_zoomed = False
        self.dragging_line = False
        
        # tomography variables
        self.shifts = [[0, 0] for _ in range(self.n_proj)] # list of [y_shift, x_shift]
        self.tomo_zoomed_size = self.raw_size // 2
        self.tomo_vertex = self._get_tomo_zoomed_vertex()

        # horizontal sum variables
        self.hs_array = self._get_hs_array()

        # sinogram variables
        self.sino_crop = (0, self.raw_size, 0, self.n_proj)  # y0, y1, x0, x1
        self.sino_selecting = False
        self.sino_arr_start = None
        self.sino_pix_start = None

        # initialize GUI
        self._init_labels()
        self._init_buttons()
        self._init_slider()
        self._setup_layout()
        self.img_label.installEventFilter(self)
        self.sino_label.installEventFilter(self)
        self.resize_view()

    def _init_labels(self):
        label_style = """
            QLabel {
                border: 2px solid #bfc7d5;
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #f5f7fa, stop:1 #c3cfe2);
                padding: 4px;
            }
        """
        self.img_label = QLabel("Tomography")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_label.setStyleSheet(label_style)

        self.hs_label = QLabel("Horizontal Sum")
        self.hs_label.setAlignment(Qt.AlignCenter)
        self.hs_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.hs_label.setStyleSheet(label_style)

        self.sino_label = QLabel("Sinogram")
        self.sino_label.setAlignment(Qt.AlignCenter)
        self.sino_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sino_label.setStyleSheet(label_style)

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
        self.hs_align_btn = QPushButton('HS Align') 
        self.hs_align_btn.setToolTip('Open auto-alignment dialog (Cross-correlation)')
        self.done_btn = QPushButton('Finish')
        self.change_center_btn = QPushButton('Change center')
        self.change_center_btn.setToolTip('Next click: Change rotational center')
        self.zoom_tomo_btn = QPushButton('Zoom tomo')
        self.reset_sino_btn = QPushButton('Reset sino')
        self.reset_sino_btn.setToolTip('Reset sinogram view')

        for btn in [self.prev_btn, self.next_btn, self.save_btn, self.load_btn, self.hs_align_btn,
                    self.done_btn, self.change_center_btn, self.zoom_tomo_btn, self.reset_sino_btn]:
            btn.setFont(self.FONT_CTRL)

        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.save_btn.clicked.connect(self.save_shifts)
        self.load_btn.clicked.connect(self.load_shifts)
        self.hs_align_btn.clicked.connect(self.open_auto_align_dialog)
        self.done_btn.clicked.connect(self.finish)
        self.change_center_btn.clicked.connect(self.start_change_center)
        self.zoom_tomo_btn.clicked.connect(self.toggle_zoom_tomo)
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
        zoom_hbox.addWidget(self.change_center_btn)
        zoom_hbox.addWidget(self.zoom_tomo_btn)
        zoom_hbox.addStretch(1)
        zoom_hbox.addWidget(self.reset_sino_btn)
        zoom_hbox.addWidget(self.zoom_in_btn)
        zoom_hbox.addWidget(self.zoom_out_btn)

        # images (tomo & sino) with reset button under sino
        tomo_vbox = QVBoxLayout()
        tomo_title = QLabel("Tomography")
        tomo_title.setFont(QFont('Calibri', 16, QFont.Bold))
        tomo_title.setAlignment(Qt.AlignCenter)
        tomo_vbox.addWidget(tomo_title)
        tomo_vbox.addWidget(self.img_label)

        hs_vbox = QVBoxLayout()
        hs_title = QLabel("Horizontal Sum")
        hs_title.setFont(QFont('Calibri', 16, QFont.Bold))
        hs_title.setAlignment(Qt.AlignCenter)
        hs_vbox.addWidget(hs_title)
        hs_vbox.addWidget(self.hs_label)

        sino_vbox = QVBoxLayout()
        sino_title = QLabel("Sinogram")
        sino_title.setFont(QFont('Calibri', 16, QFont.Bold))
        sino_title.setAlignment(Qt.AlignCenter)
        sino_vbox.addWidget(sino_title)
        sino_vbox.addWidget(self.sino_label)

        top_hbox = QHBoxLayout()
        top_hbox.setSpacing(20)
        top_hbox.addLayout(tomo_vbox)
        top_hbox.addLayout(hs_vbox)
        top_hbox.addLayout(sino_vbox)

        layout = QVBoxLayout()
        layout.addLayout(zoom_hbox)
        layout.addLayout(top_hbox)
        # control buttons (without reset)
        hbox = QHBoxLayout()
        hbox.addWidget(self.prev_btn)
        hbox.addWidget(self.next_btn)
        hbox.addWidget(self.save_btn)
        hbox.addWidget(self.load_btn)
        hbox.addWidget(self.hs_align_btn)
        hbox.addWidget(self.done_btn)
        layout.addLayout(hbox)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def start_change_center(self):
        self.changing_center = True

    def toggle_zoom_tomo(self):
        self.tomo_zoomed = not self.tomo_zoomed
        self.update_tomo()
        self.hs_array = self._get_hs_array()
        self.update_hori_sum()

    def reset_sino_view(self):
        self.sino_crop = (0, self.raw_size, 0, self.n_proj)
        self.sino_selecting = False
        self.sino_arr_start = None
        self.sino_pix_start = None
        self.update_sino()

    def zoom_in(self):
        if (self.view_scale * 1.25) < self.max_scale:
            self.view_scale *= 1.25
            self.resize_view()

    def zoom_out(self):
        if (self.view_scale / 1.25) > self.min_scale:
            self.view_scale /= 1.25
            self.resize_view()

    def resize_view(self):
        base_w, base_h = 1600, 900
        new_w = int(base_w * self.view_scale)
        new_h = int(base_h * self.view_scale)
        self.setFixedSize(new_w, new_h)

        # distribute space of three labels
        total_w = self.width() - 40  
        total_h = self.height() - 180  
        img_label_w = int(total_w * 0.5)
        hs_label_w = int(total_w * 0.25) 
        sino_label_w = total_w - img_label_w - hs_label_w
        img_label_h = hs_label_h = sino_label_h = total_h
        img_label_size = min(img_label_w, img_label_h)
        self.img_label.setFixedSize(img_label_size, img_label_size)
        self.hs_label.setFixedSize(hs_label_w, hs_label_h)
        self.sino_label.setFixedSize(sino_label_w, sino_label_h)
        scale_w = img_label_w / self.raw_size
        scale_h = img_label_h / self.raw_size
        self.scale = min(scale_w, scale_h)

        self.update_all()

    def slider_changed(self, value):
        self.index = value
        self.update_all()
        self.setWindowTitle(f"Align Viewer {self.index + 1}/{self.n_proj}")

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
                for i, (dy, dx) in enumerate(self.shifts):
                    f.write(f"{str(i).zfill(3)},{dy},{dx}\n")

    def load_shifts(self):
        shifts_file, _ = QFileDialog.getOpenFileName(None, "Load Shifts", "*.txt")
        if shifts_file:
            with open(shifts_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    try:
                        idx, dy, dx = map(int, parts)
                        self.shifts[idx] = [dy, dx]  
                        self.proj_images[idx] = np.roll(self.proj_images[idx], shift=(dy, dx), axis=(0, 1))
                    except Exception:
                        continue
        self.update_all()

    def open_auto_align_dialog(self):
        features = self.hs_array.copy()
        
        dlg = CCAlignDialog(features, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            calculated_shifts = dlg.get_shifts()
            self._apply_cc_shifts(calculated_shifts)

    def finish(self):
        for i in range(self.n_proj):
            img = self.tomo.get_image(i)
            img = np.roll(img, shift=self.shifts[i], axis=(0, 1))
            self.tomo.set(i, img)
        super().accept()

    def update_all(self):
        self.update_tomo()
        self.update_sino()
        self.update_hori_sum()

    # -------------- core logic --------------- 
    def _apply_cc_shifts(self, y_shifts):
        for i, dy in enumerate(y_shifts):
            self.shifts[i][0] += dy 
            self.proj_images[i] = np.roll(self.proj_images[i], shift=dy, axis=0)
            self.hs_array[:, i] = np.roll(self.hs_array[:, i], shift=dy)
            
        self.update_all()

    def _get_tomo_zoomed_vertex(self):
        center_x, center_y = self.rotational_center
        x0 = center_x - self.tomo_zoomed_size // 2
        y0 = max(center_y - self.tomo_zoomed_size // 2, 0) 
        y1 = min(y0 + self.tomo_zoomed_size, self.raw_size)
        if y1 - y0 < self.tomo_zoomed_size:
            y0 = y1 - self.tomo_zoomed_size
        return x0, y0

    def _get_hs_array(self):
        """
        Get horizontal sum array, possibly cropped around vertex.

        Returns
        -------
        hs_array : np.ndarray
            The horizontal sum array in 8-bit format.
        """
        if not self.tomo_zoomed:
            hs_array = self.proj_images.sum(axis=2)
        else:
            x0, y0 = self.tomo_vertex
            crop_size = self.raw_size // 2
            x1 = x0 + crop_size
            y1 = y0 + crop_size
            hs_array = self.proj_images[:, y0:y1, x0:x1].sum(axis=2)
        return norm_hs_to_8bit(hs_array).T

    def _get_sino_pos(self, event):
        """
        get the actual sino position when clicking on the sino image.

        Returns
        -------
        sino_x, sino_y : int
            x and y coordinates in the sinogram array.
        """
        y0, y1, x0, x1 = self.sino_crop
        click_x, click_y = event.x(), event.y()
        pm_w, pm_h = self.sino_label.pixmap().width(), self.sino_label.pixmap().height()
        click_x = max(0, min(click_x, pm_w))
        click_y = max(0, min(click_y, pm_h))
        sino_h, sino_w = y1 - y0, x1 - x0
        sino_x = x0 + int(click_x * sino_w / pm_w)
        sino_y = y0 + int(click_y * sino_h / pm_h)
        return sino_x, sino_y

    def _set_label_pixmap(self, label, img_array, keep_ratio=True):
        img_array = np.ascontiguousarray(img_array)
        h, w = img_array.shape[:2]
        rect = label.contentsRect()
        label_w, label_h = rect.width(), rect.height()
        
        if img_array.ndim == 2:
            qimg = QImage(img_array.data, w, h, w, QImage.Format_Grayscale8)
        else:
            qimg = QImage(img_array.data, w, h, w*3, QImage.Format_RGB888)
        
        if keep_ratio:
            pixmap = QPixmap.fromImage(qimg).scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            pixmap = QPixmap.fromImage(qimg).scaled(label_w, label_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        """
        Use WASD keys to shift the tomography image.
        W/S: shift up/down
        A/D: shift left/right
        """
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
            img_temp = self.proj_images[self.index]
            self.proj_images[self.index] = np.roll(img_temp, shift=delta, axis=axis)
            if axis == 0:
                self.hs_array[:, self.index] = np.roll(self.hs_array[:, self.index], shift=delta)
            self.update_all()

    def eventFilter(self, obj, event):
        # sinogram event filter
        if obj is self.sino_label:
            # start selecting
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                coord = self._get_sino_pos(event.pos())
                self.sino_selecting = True
                self.sino_arr_start = coord
                self.sino_pix_start = event.pos().x(), event.pos().y()
                return True
            # moving mouse while selecting
            elif event.type() == QEvent.MouseMove and self.sino_selecting:
                coord = self._get_sino_pos(event.pos())
                if self.sino_pix_start is not None:
                    x0, y0 = self.sino_pix_start
                    x1, y1 = event.pos().x(), event.pos().y()
                    self.update_sino(draw_rect=(min(x0, x1), min(y0, y1),
                                                max(x0, x1), max(y0, y1)))
                return True
            # finish selecting
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self.sino_selecting and self.sino_pix_start is not None:
                    coord = self._get_sino_pos(event.pos())
                    x0, y0 = self.sino_arr_start
                    x1, y1 = coord
                    if (x0, y0) != (x1, y1):
                        self.sino_crop = (min(y0, y1), max(y0, y1),
                                            min(x0, x1), max(x0, x1))
                    self.update_sino()
                self.sino_selecting = False
                self.sino_pix_start = None
                return True
            
        # tomography event filter
        if obj is self.img_label:
            if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
                img_x = int(event.x() / self.scale)
                img_y = int(event.y() / self.scale)

                # prevent out-of-bounds
                if img_x < 0 or img_x >= self.raw_size or img_y < 0 or img_y >= self.raw_size:
                    return True
                # change the coordinates if zoomed
                if self.tomo_zoomed:
                    img_x, img_y = img_x//2, img_y//2
                    img_x += self.tomo_vertex[0]
                    img_y += self.tomo_vertex[1]
                
                if self.changing_center:
                    self.rotational_center = (self.raw_size // 2, img_y)
                    self.changing_center = False
                    self.tomo_vertex = self._get_tomo_zoomed_vertex()
                    self.update_tomo()
                    return True
                # click on green line
                if abs(img_y - self.line_y) < 5:
                    self.dragging_line = True
                    return True
                # click on tomo image
                else:
                    dx = self.rotational_center[0] - img_x
                    dy = self.rotational_center[1] - img_y
                    self.shifts[self.index][0] += dy
                    self.shifts[self.index][1] += dx 
                    self.proj_images[self.index] = np.roll(self.proj_images[self.index], shift=(dy, dx), axis=(0, 1))
                    self.hs_array[:, self.index] = np.roll(self.hs_array[:, self.index], shift=dy)
                    self.update_all()
                    return True
            # moving mouse while dragging line
            elif event.type() == QEvent.MouseMove and self.dragging_line:
                img_y = int(event.y() / self.scale)
                img_y = max(0, min(img_y, self.raw_size - 1))
                if self.tomo_zoomed:
                    img_y = img_y//2 + self.tomo_vertex[1]
                self.line_y = img_y
                self.update_tomo()
                self.update_sino()
                return True
            # finish dragging line
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.dragging_line = False
                return True
            
        return super().eventFilter(obj, event)

    def update_tomo(self):
        size = self.raw_size
        center_x, center_y = self.rotational_center
        base_img = self.proj_images[self.index]
        base_img = Image.fromarray(base_img).convert("RGBA")
        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        
        # support line
        line_scale = 512 // size
        draw.line([(0, self.line_y), (base_img.width, self.line_y)], fill=self.line_color, width=1)
        # crosshair at rotational center
        r = 4 / line_scale
        draw.line([(center_x - r, center_y), (center_x + r, center_y)], fill=self.line_color, width=2//line_scale)
        draw.line([(center_x, center_y - r), (center_x, center_y + r)], fill=self.line_color, width=2//line_scale)
        result_img = Image.alpha_composite(base_img, overlay)
        img_rgb = np.array(result_img.convert("RGB"))

        if self.tomo_zoomed:
            x0, y0 = self.tomo_vertex
            x1 = x0 + self.tomo_zoomed_size
            y1 = y0 + self.tomo_zoomed_size
            img_rgb = img_rgb[y0:y1, x0:x1, :].copy()

        self._set_label_pixmap(self.img_label, img_rgb, keep_ratio=True)

    def update_hori_sum(self):
        hs_8bit = self.hs_array.copy()
        self._set_label_pixmap(self.hs_label, hs_8bit, keep_ratio=False)

        # draw current index arrow
        pixmap = self.hs_label.pixmap()
        rect = self.hs_label.contentsRect()
        label_w, label_h = rect.width(), rect.height()
        arrow_x = self.index 
        if 0 <= arrow_x < hs_8bit.shape[1]:
            painter = QPainter(pixmap)
            pen = QPen(QColor(255, 0, 0, 128), 5)
            painter.setPen(pen)
            arrow_x_pixmap = int(arrow_x * label_w / hs_8bit.shape[1]) + 1 
            segment_h = int(label_h * 0.05)
            painter.drawLine(arrow_x_pixmap, 0, arrow_x_pixmap, segment_h)  
            painter.drawLine(arrow_x_pixmap, label_h - segment_h, arrow_x_pixmap, label_h) 
            painter.end()

    def update_sino(self, draw_rect=None):
        """
        Parameters
        ----------
        draw_rect : tuple or None
            If provided, should be (x0, y0, x1, y1) to draw a rectangle on the sinogram preview.
        """
        row_index = int(self.line_y)
        sino = self.proj_images[:, row_index, :].T  # shape: (width, N_proj)
        # crop sinogram
        y0, y1, x0, x1 = self.sino_crop
        sino = sino[y0:y1, x0:x1]
        # show image
        sino_rgb = np.repeat(sino[:, :, None], 3, axis=2)
        self._set_label_pixmap(self.sino_label, sino_rgb, keep_ratio=False)

        # draw current index arrow
        pixmap = self.sino_label.pixmap()
        rect = self.sino_label.contentsRect()
        label_w, label_h = rect.width(), rect.height()
        arrow_x = self.index - x0 
        if 0 <= arrow_x < sino_rgb.shape[1]:
            painter = QPainter(pixmap)
            pen = QPen(QColor(255, 0, 0, 128), 5)
            painter.setPen(pen)
            arrow_x_pixmap = int(arrow_x * label_w / sino_rgb.shape[1]) + 1 
            segment_h = int(label_h * 0.05)
            painter.drawLine(arrow_x_pixmap, 0, arrow_x_pixmap, segment_h)  
            painter.drawLine(arrow_x_pixmap, label_h - segment_h, arrow_x_pixmap, label_h) 
            painter.end()

        # draw selection rectangle
        if draw_rect is not None:
            rx0, ry0, rx1, ry1 = draw_rect
            painter = QPainter(pixmap)
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.drawLine(rx0, ry0, rx1, ry0)  
            painter.drawLine(rx0, ry1, rx1, ry1)  
            painter.drawLine(rx0, ry0, rx0, ry1)  
            painter.drawLine(rx1, ry0, rx1, ry1) 
            painter.end()
