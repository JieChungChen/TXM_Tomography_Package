import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from src.logic.utils import norm_to_8bit


class DuplicateAngleResolver(QDialog):
    def __init__(self, images, theta_value, ref, file_names):
        super().__init__()
        self.setWindowTitle(f"Select best image for θ = {theta_value:.2f}")
        self.selected_idx = None

        layout = QVBoxLayout()
        hbox = QHBoxLayout()

        for i, (img, fname) in enumerate(zip(images, file_names)):
            label = QLabel()
            img = img / ref
            img8 = norm_to_8bit(img)
            h, w = img8.shape
            qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg).scaled(400, 400, Qt.KeepAspectRatio)

            label.setPixmap(pixmap)
            label.mousePressEvent = lambda e, idx=i: self.select(idx)

            # 檔案名label
            fname_label = QLabel(fname)
            fname_label.setStyleSheet("font-family: Calibri; font-size: 10pt; color: red; text-align: center;")

            # 垂直佈局：影像在上，檔案名在下
            vbox = QVBoxLayout()
            vbox.addWidget(label)
            vbox.addWidget(fname_label)
            hbox.addLayout(vbox)

        layout.addLayout(hbox)
        self.setLayout(layout)

    def select(self, idx):
        self.selected_idx = idx
        self.accept()

    def get_selection(self):
        return self.selected_idx


def resolve_duplicates(images, thetas, duplicates, ref, file_names):
    selected_indices = set()  # 收集選中索引

    # 處理重複組
    for group in duplicates:
        imgs = [images[i] for i in group]
        theta_val = thetas[group[0]]
        group_file_names = [file_names[i] for i in group]

        dialog = DuplicateAngleResolver(imgs, theta_val, ref, group_file_names)
        if dialog.exec_() == QDialog.Accepted:
            idx = dialog.get_selection()
            chosen_idx = group[idx]
            selected_indices.add(chosen_idx)

    # 添加非重複索引（移除 is_integer 條件，保留所有）
    all_idx = set(range(len(thetas)))
    dup_idx = set(i for group in duplicates for i in group)
    nondup_idx = all_idx - dup_idx
    selected_indices.update(nondup_idx)

    # 篩選並按照角度排序
    selected_images = images[list(selected_indices)]
    selected_thetas = thetas[list(selected_indices)]
    sort_idx = np.argsort(selected_thetas)
    selected_images = selected_images[sort_idx]
    selected_thetas = selected_thetas[sort_idx]

    return selected_images, selected_thetas