import sys
import os
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QComboBox, QScrollArea, QMessageBox, QStatusBar,
    QProgressBar, QFrame, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

from ultralytics import YOLO

class VideoWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, dict)  # 帧与统计信息

    def __init__(self, model, video_source=0, conf_threshold=0.3):
        super().__init__()
        self.model = model
        self.video_source = video_source
        self.conf_threshold = conf_threshold
        self.running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.video_source)
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            results = self.model(frame, conf=self.conf_threshold, device='cpu')
            annotated_frame = results[0].plot()
            stats = self.extract_stats(results[0])
            self.frame_ready.emit(annotated_frame, stats)

        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def extract_stats(self, result):
        counts = {}
        total_conf = 0
        for box in result.boxes:
            cls = int(box.cls)
            name = self.model.names[cls]
            counts[name] = counts.get(name, 0) + 1
            total_conf += float(box.conf)

        avg_conf = total_conf / len(result.boxes) if result.boxes else 0
        return {
            "total_objects": sum(counts.values()),
            "class_distribution": counts,
            "avg_confidence": avg_conf,
            "fps": getattr(self, "fps", 0),
        }

class YOLODetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水稻害虫智能检测系统 - PyQt5版")
        self.resize(1400, 900)

        # 初始化变量
        self.model = None
        self.model_path = ""
        self.save_dir = "results"
        self.worker = None
        self.dark_theme = True  # 默认暗黑模式

        self.init_ui()
        self.init_status_bar()
        self.apply_theme()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # 左侧控制面板
        control_panel = self.create_control_panel()
        control_panel.setFixedWidth(320)
        main_layout.addWidget(control_panel)

        # 右侧图像与信息展示区
        right_layout = QVBoxLayout()

        # 图像显示区
        self.image_scroll = QScrollArea()
        self.image_label = QLabel("等待检测...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e; border-radius: 10px;")
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setWidgetResizable(True)
        right_layout.addWidget(self.image_scroll)

        # 检测统计信息区
        self.stats_card = self.create_stats_card()
        right_layout.addWidget(self.stats_card)

        main_layout.addLayout(right_layout, 2)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def create_stats_card(self):
        """创建检测统计信息卡片"""
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet("""
            QFrame#card {
                background-color: #3a3a3a;
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
            }
            QLabel.cardTitle {
                color: #ffffff;
                font-weight: bold;
                font-size: 18px;
                margin-bottom: 10px;
            }
            QLabel.statsText {
                font-size: 14px;
                line-height: 1.5;
            }
        """)
        layout = QVBoxLayout()
        layout.addWidget(self.create_label("检测统计", "cardTitle"))

        self.stats_text = QLabel("暂无检测数据\n\n- 加载模型并开始检测以查看统计信息\n- 统计信息包括：类别数量、平均置信度等")
        self.stats_text.setObjectName("statsText")
        layout.addWidget(self.stats_text)
        card.setLayout(layout)
        return card

    def create_control_panel(self):
        panel = QWidget()
        panel.setObjectName("controlPanel")
        panel.setStyleSheet("""
            QWidget#controlPanel {
                background-color: #2d2d2d;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        layout = QVBoxLayout()

        # 模型加载卡片
        model_card = self.create_card("模型设置", [
            self.create_label("当前模型: 未加载", "modelLabel"),
            self.create_button("📁 加载YOLO模型", self.load_model, "primary", "modelBtn")
        ])
        layout.addWidget(model_card)

        # 输入源卡片
        input_card = self.create_card("输入源设置", [
            self.create_label("输入源选择:"),
            self.create_combo_box(["📸 摄像头", "🎞️ 视频文件", "🖼️ 图片文件"], "inputCombo"),
            self.create_button("📂 选择文件", self.select_input_file, "secondary", "fileBtn")
        ])
        layout.addWidget(input_card)

        # 控制面板卡片
        control_card = self.create_card("控制面板", [
            self.create_button("▶ 开始检测", self.start_detection, "primary", "startBtn"),
            self.create_button("⏹ 停止检测", self.stop_detection, "danger", "stopBtn"),
            self.create_button("💾 保存结果", self.save_result, "success", "saveBtn")
        ])
        layout.addWidget(control_card)

        # 主题设置卡片
        theme_card = self.create_card("主题设置", [
            self.create_label("界面主题"),
            self.create_combo_box(["🌙 暗黑模式", "☀️ 亮白模式"], "themeCombo", self.change_theme)
        ])
        layout.addWidget(theme_card)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_card(self, title, widgets):
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet("""
            QFrame#card {
                background-color: #3a3a3a;
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 10px;
            }
            QLabel.cardTitle {
                color: #ffffff;
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 5px;
            }
        """)
        layout = QVBoxLayout()
        layout.addWidget(self.create_label(title, "cardTitle"))
        for widget in widgets:
            layout.addWidget(widget)
        card.setLayout(layout)
        return card

    def create_label(self, text, style=""):
        label = QLabel(text)
        if style:
            label.setObjectName(style)
        return label

    def create_button(self, text, func, btn_type="default", obj_name=""):
        btn = QPushButton(text)
        if obj_name:
            btn.setObjectName(obj_name)
        style_map = {
            "primary": """
                QPushButton {
                    background-color: #007acc;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 6px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #005fa3;
                }
            """,
            "secondary": """
                QPushButton {
                    background-color: #444444;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 6px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #666666;
                }
            """,
            "danger": """
                QPushButton {
                    background-color: #e63946;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 6px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #b52e31;
                }
            """,
            "success": """
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 6px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #1e7e34;
                }
            """
        }
        btn.setStyleSheet(style_map.get(btn_type, ""))
        btn.clicked.connect(func)
        return btn

    def create_combo_box(self, items, obj_name, func=None):
        combo = QComboBox()
        combo.setObjectName(obj_name)
        combo.addItems(items)
        if func:
            combo.currentIndexChanged.connect(func)
        combo.setStyleSheet("""
            QComboBox {
                padding: 6px;
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #444;
                color: white;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: 0px;
            }
        """)
        return combo

    def init_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.hide()

        self.status_bar.addWidget(self.status_label, 1)
        self.status_bar.addWidget(self.progress_bar, 0)

    def load_model(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择YOLO权重文件", "", "权重文件 (*.pt)"
        )
        if not file_name:
            self.show_message("提示", "未选择任何文件。", is_error=False)
            return

        try:
            self.progress_bar.show()
            self.status_label.setText("正在加载模型...")

            # 强制使用 CPU
            self.model = YOLO(file_name).to('cpu')

            self.model_path = file_name
            self.findChild(QLabel, "modelLabel").setText(f"当前模型: {os.path.basename(file_name)}")

            self.status_label.setText("模型加载成功")
            self.progress_bar.hide()
            self.show_message("成功", "模型加载完成", is_error=False)

        except FileNotFoundError:
            self.status_label.setText("模型加载失败 - 文件未找到")
            self.progress_bar.hide()
            self.show_message("错误", "文件未找到，请确认路径是否正确。", is_error=True)

        except Exception as e:
            self.status_label.setText("模型加载失败 - 未知错误")
            self.progress_bar.hide()
            error_msg = str(e)
            detailed_msg = f"加载模型时发生错误:\n\n{error_msg}\n\n请检查以下内容:\n1. 是否是有效的 YOLOv8 模型文件(.pt)\n2. 是否为最新 ultralytics 版本\n3. 是否缺少依赖库"
            self.show_message("加载失败", detailed_msg, is_error=True)

    def select_input_file(self):
        input_combo = self.findChild(QComboBox, "inputCombo")
        if input_combo.currentIndex() == 1:
            self.file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)"
            )
        elif input_combo.currentIndex() == 2:
            self.file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片文件", "", "图片文件 (*.png *.jpg *.bmp)"
            )

    def start_detection(self):
        if not self.model:
            self.show_message("警告", "请先加载模型！", is_error=True)
            return

        source_type = self.findChild(QComboBox, "inputCombo").currentIndex()
        if source_type == 2:
            self.process_single_image()
            return

        self.findChild(QPushButton, "startBtn").setEnabled(False)
        self.findChild(QPushButton, "stopBtn").setEnabled(True)

        self.status_label.setText("检测进行中")

        video_source = 0 if source_type == 0 else self.file_path
        self.worker = VideoWorker(self.model, video_source, conf_threshold=0.3)
        self.worker.frame_ready.connect(self.display_image_and_stats)
        self.worker.start()

    def process_single_image(self):
        if not hasattr(self, "file_path") or not os.path.exists(self.file_path):
            self.show_message("警告", "请选择有效的图片文件！", is_error=True)
            return

        img = cv2.imread(self.file_path)
        results = self.model(img, conf=0.3)
        annotated_img = results[0].plot()
        self.display_image(annotated_img)
        self.update_stats(results[0])

    def display_image_and_stats(self, frame, stats):
        self.display_image(frame)
        self.update_stats(stats)

    def display_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaledToWidth(
            self.image_scroll.width() - 20,
            Qt.SmoothTransformation
        ))

    def update_stats(self, stats):
        if isinstance(stats, dict):
            stats_text = f"当前帧检测到: {stats.get('total_objects', 0)} 个对象\n\n"
            stats_text += "类别分布:\n" + "\n".join([f"  • {k}: {v} 个" for k, v in stats.get('class_distribution', {}).items()])
            stats_text += f"\n\n平均置信度: {stats.get('avg_confidence', 0):.2f}"
            self.stats_text.setText(stats_text)

    def stop_detection(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.findChild(QPushButton, "startBtn").setEnabled(True)
        self.findChild(QPushButton, "stopBtn").setEnabled(False)
        self.status_label.setText("已停止检测")

    def save_result(self):
        if not hasattr(self, "last_image"):
            self.show_message("警告", "没有可保存的结果！", is_error=True)
            return

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", os.path.join(self.save_dir, "result.jpg"), "图像 (*.jpg)"
        )
        if file_name:
            cv2.imwrite(file_name, self.last_image)
            self.show_message("保存成功", f"结果已保存至: {file_name}")

    def change_theme(self, index):
        self.dark_theme = index == 0
        self.apply_theme()
        self.show_message("主题已切换", f"当前主题: {'暗黑' if index == 0 else '亮白'}")

    def apply_theme(self):
        if self.dark_theme:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1e1e1e;
                    color: white;
                }
                QLabel {
                    color: white;
                }
                QComboBox, QSlider, QPushButton {
                    border: 1px solid #555;
                    padding: 6px;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f0f0f0;
                    color: black;
                }
                QLabel {
                    color: black;
                }
                QComboBox, QSlider, QPushButton {
                    border: 1px solid #ccc;
                    padding: 6px;
                }
            """)

    def show_message(self, title, content, is_error=False):
        if is_error:
            QMessageBox.critical(self, title, content)
        else:
            QMessageBox.information(self, title, content)

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("微软雅黑", 10))
    window = YOLODetector()
    window.show()
    sys.exit(app.exec_())