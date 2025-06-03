import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QMessageBox, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

# 英文标签到中文名称的映射
LABEL_MAP = {
    'green-leafhopper': '绿色叶蝉',
    'rice-bug': '稻蝽',
    'leaf-folder': '卷叶虫',
    'stem-borer': '茎螟',
    'whorl-maggot': '叶鞘蛆'
}

# 农药数据库（英文标签: [农药名称, 单位用量(ml/亩)]）
PESTICIDE_DB = {
    'green-leafhopper': ["吡虫啉", 30],
    'rice-bug': ["氯氰菊酯", 50],
    'leaf-folder': ["甲维盐", 40],
    'stem-borer': ["氯虫苯甲酰胺", 60],
    'whorl-maggot': ["吡蚜酮", 35]
}

class SimplePesticideApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("防治水稻害虫农药推荐系统")
        self.resize(800, 600)

        self.model = None
        self.current_img = None
        self.pest_data = []  # 存储结构化的害虫数据

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QHBoxLayout()

        # 左侧面板
        left_layout = QVBoxLayout()

        self.load_model_btn = QPushButton("📁 加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        left_layout.addWidget(self.load_model_btn)

        self.select_image_btn = QPushButton("📂 选择图片")
        self.select_image_btn.clicked.connect(self.select_image)
        self.select_image_btn.setEnabled(False)
        left_layout.addWidget(self.select_image_btn)

        self.area_input = QLineEdit()
        self.area_input.setPlaceholderText("请输入田地面积（亩）")
        self.area_input.textChanged.connect(self.update_recommendation)
        left_layout.addWidget(self.area_input)

        self.run_detection_btn = QPushButton("🔍 开始检测")
        self.run_detection_btn.clicked.connect(self.run_detection)
        self.run_detection_btn.setEnabled(False)
        left_layout.addWidget(self.run_detection_btn)

        self.result_label = QLabel("检测结果与推荐农药将显示在此")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("padding: 10px;")
        self.result_label.setMinimumHeight(200)
        left_layout.addWidget(self.result_label)

        layout.addLayout(left_layout, 1)

        # 右侧图像预览
        self.image_label = QLabel("图像预览区域")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ddd; padding: 10px;")
        layout.addWidget(self.image_label, 2)

        main_widget.setLayout(layout)

    def load_model(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择YOLO模型", "", "模型文件 (*.pt)"
        )
        if not file_name:
            return

        try:
            self.model = YOLO(file_name)
            self.select_image_btn.setEnabled(True)
            self.statusBar().showMessage("模型加载成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败：{str(e)}")

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "", "图片文件 (*.jpg *.png *.bmp)"
        )
        if not file_name:
            return

        try:
            self.current_img = cv2.imread(file_name)
            self.display_image(self.current_img)
            self.run_detection_btn.setEnabled(True)
            self.statusBar().showMessage("图片加载成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图片失败：{str(e)}")

    def display_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        bytes_per_line = 3 * w
        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaledToWidth(500))

    def run_detection(self):
        if not self.model or self.current_img is None:
            return

        try:
            results = self.model(self.current_img)
            annotated_img = results[0].plot()
            self.display_image(annotated_img)

            # 清空旧数据
            self.pest_data.clear()

            # 统计害虫数据（结构化存储）
            for box in results[0].boxes:
                cls = int(box.cls)
                eng_label = self.model.names[cls]
                chi_name = LABEL_MAP.get(eng_label, eng_label)
                pesticide, dosage = PESTICIDE_DB.get(eng_label, ["未知", 0])
                self.pest_data.append({
                    "english": eng_label,
                    "chinese": chi_name,
                    "pesticide": pesticide,
                    "base_dosage": dosage,
                    "count": 1
                })

            # 聚合相同害虫
            pest_dict = {}
            for item in self.pest_data:
                key = item["english"]
                if key in pest_dict:
                    pest_dict[key]["count"] += 1
                else:
                    pest_dict[key] = item.copy()

            self.pest_data = list(pest_dict.values())

            self.update_recommendation()
            self.statusBar().showMessage("检测完成")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败：{str(e)}")

    def update_recommendation(self):
        area_text = self.area_input.text()
        try:
            area = float(area_text) if area_text else 1
        except:
            area = 1

        if not self.pest_data:
            self.result_label.setText("未检测到害虫")
            return

        result_lines = []
        for item in self.pest_data:
            chi_name = item["chinese"]
            pesticide = item["pesticide"]
            base_dosage = item["base_dosage"]
            total_dosage = base_dosage * area
            result_lines.append(f"【{chi_name}】")
            result_lines.append(f"推荐农药：{pesticide}")
            result_lines.append(f"基础用量：{base_dosage}ml/亩")
            result_lines.append(f"总用量：{total_dosage:.1f}ml")
            result_lines.append("-" * 20)

        self.result_label.setText("\n".join(result_lines))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimplePesticideApp()
    window.show()
    sys.exit(app.exec_())