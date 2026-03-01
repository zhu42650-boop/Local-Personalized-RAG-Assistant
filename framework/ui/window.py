import os
import threading
import copy
from datetime import datetime
from typing import Iterable, List

from PySide6 import QtCore, QtGui, QtWidgets
from langchain_openai import ChatOpenAI

from config.env_check import ensure_dirs
from config.loader import load_settings, resolve_paths
from ingest.file_manager import add_files_to_category
from ingest.service import run_ingest
from rag.chat import answer_question
from rag.retriever import get_retriever

# --- 全局样式表 (QSS) ---
STYLES = """
QMainWindow {
    background-color: #F0F2F5;
}
QTextEdit {
    background-color: #F0F2F5;
    border: none;
}
QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 5px;
    padding: 12px;
    font-size: 15px;
    color: #333;
}
QLineEdit:focus {
    border: 1px solid #007AFF;
}
/* 发送按钮 */
QPushButton#sendBtn {
    background-color: #007AFF;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 8px 20px;
    font-weight: bold;
}
QPushButton#sendBtn:hover {
    background-color: #0062CC;
}
/* 状态栏 */
QLabel#statusBar {
    color: #999;
    font-size: 12px;
    padding: 5px 10px;
}
/* 拖拽面板 */
QFrame#DropPanel {
    background-color: rgba(255, 255, 255, 0.8);
    border: 2px dashed #B0B0B0;
    border-radius: 10px;
}
QFrame#DropPanel:hover {
    background-color: rgba(255, 255, 255, 1.0);
    border-color: #007AFF;
}
"""

class UiSignals(QtCore.QObject):
    append_chat = QtCore.Signal(str, str)
    set_status = QtCore.Signal(str)

class DropPanel(QtWidgets.QFrame):
    def __init__(self, title: str, color_hex: str, category: str, on_files):
        super().__init__()
        self.setObjectName("DropPanel") # 用于QSS定位
        self.category = category
        self.on_files = on_files
        self.setAcceptDrops(True)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # 标题
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet(f"color: {color_hex}; font-size: 16px; font-weight: bold;")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # 提示文字
        hint = QtWidgets.QLabel("拖拽文件至此")
        hint.setStyleSheet("color: #888; font-size: 12px;")
        hint.setAlignment(QtCore.Qt.AlignCenter)
        
        layout.addStretch(1)
        layout.addWidget(title_label)
        layout.addWidget(hint)
        layout.addStretch(1)

        # 调整边框颜色以匹配类别
        self.setStyleSheet(f"""
            QFrame#DropPanel {{
                border: 2px dashed {color_hex};
                background-color: {color_hex}10;
            }}
        """)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        files = [f for f in files if f]
        if files:
            self.on_files(self.category, files)
        event.acceptProposedAction()


class BubbleWidget(QtWidgets.QWidget):
    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.text = text
        
        # 布局初始化
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标签初始化
        self.label = QtWidgets.QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        
        # 字体设置 (保持与全局字体一致，防止计算误差)
        font = QtGui.QFont("Microsoft YaHei", 12)  # 或者你在 launch_ui 里设置的字体
        self.label.setFont(font)
        
        # 样式：增加 padding 让文字不贴边
        self.label.setStyleSheet("color:#333; padding: 12px 14px;")
        
        layout.addWidget(self.label)

        # --- 核心修复逻辑：动态计算宽度 ---
        # 1. 获取字体测量工具
        fm = QtGui.QFontMetrics(font)
        
        # 2. 计算文字单行显示的理论宽度
        # boundingRect 能够计算出文字在屏幕上的像素矩形
        text_rect = fm.boundingRect(QtCore.QRect(0, 0, 0, 0), QtCore.Qt.AlignCenter, text)
        text_width = text_rect.width()
        
        # 3. 增加额外的 padding 宽度 (对应上面 stylesheet 的 padding + 气泡边框)
        total_width = text_width + 35 
        
        # 4. 设定最大宽度限制 (例如屏幕宽度的 60% 或固定值 600)
        MAX_WIDTH = 650
        
        # 5. 决策：如果文字短，用实际宽度；如果文字长，卡在最大宽度
        final_width = min(total_width, MAX_WIDTH)
        
        # 6. 加上最小宽度限制，防止只有“嗨”字时气泡太圆
        final_width = max(final_width, 60)
        
        # 强制设置固定宽度 (关键步骤)
        self.setFixedWidth(final_width)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # 绘制区域：基于整个 Widget 的大小
        rect = self.rect().adjusted(1, 1, -1, -1)
        
        radius = 10
        # 颜色配置
        if self.is_user:
            bg_color = QtGui.QColor("#95EC69") # 微信绿风格，比原来的蓝更柔和
            border_color = QtGui.QColor("#85D65D")
        else:
            bg_color = QtGui.QColor("#FFFFFF")
            border_color = QtGui.QColor("#E0E0E0")
            
        painter.setBrush(bg_color)
        painter.setPen(border_color)
        
        # 绘制圆角矩形
        painter.drawRoundedRect(rect, radius, radius)
        
        # 绘制小三角 (气泡尾巴)
        arrow = QtGui.QPolygon()
        arrow_size = 6
        arrow_y = 18 # 尾巴的高度位置
        
        if self.is_user:
            # 右侧尾巴
            x = rect.right()
            arrow << QtCore.QPoint(x, arrow_y) \
                  << QtCore.QPoint(x + arrow_size, arrow_y + arrow_size) \
                  << QtCore.QPoint(x, arrow_y + arrow_size * 2)
        else:
            # 左侧尾巴
            x = rect.left()
            arrow << QtCore.QPoint(x, arrow_y) \
                  << QtCore.QPoint(x - arrow_size, arrow_y + arrow_size) \
                  << QtCore.QPoint(x, arrow_y + arrow_size * 2)
                  
        painter.drawPolygon(arrow)
        painter.end()

class ChatWindow(QtWidgets.QMainWindow):
    def __init__(self, config_path: str):
        super().__init__()
        self.settings = load_settings(config_path)
        self.paths = resolve_paths(self.settings, config_path)
        ensure_dirs(self.paths)

        self.signals = UiSignals()
        self.signals.append_chat.connect(self._append_chat)
        self.signals.set_status.connect(self._set_status)

        self.retriever = None
        self.llm = None
        self.summary_llm = None
        self.history = []
        self.current_session = []
        self.loading_history = False
        self.loaded_from_history = False
        self.session_dirty = False
        self.loaded_session_index = None
        self.history_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "history.json"
        )

        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle(self.settings.get("ui.window_title") or "RAG 知识库助手")
        self.resize(1100, 800)
        self.setStyleSheet(STYLES)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 左侧栏
        sidebar = QtWidgets.QFrame()
        sidebar.setFixedWidth(56)
        sidebar.setStyleSheet("QFrame{background:#E6ECF5;}")
        side_layout = QtWidgets.QVBoxLayout(sidebar)
        side_layout.setContentsMargins(8, 12, 8, 12)
        side_layout.setSpacing(10)

        self.history_btn = QtWidgets.QPushButton("🕘")
        self.history_btn.setToolTip("查看历史")
        self.history_btn.clicked.connect(self.on_show_history)
        self.history_btn.setFixedSize(40, 40)
        self.history_btn.setStyleSheet("border-radius:20px; font-size:16px;")

        self.newchat_btn = QtWidgets.QPushButton("＋")
        self.newchat_btn.setToolTip("新对话")
        self.newchat_btn.clicked.connect(self.on_new_chat)
        self.newchat_btn.setFixedSize(40, 40)
        self.newchat_btn.setStyleSheet("border-radius:20px; font-size:16px;")

        side_layout.addWidget(self.history_btn)
        side_layout.addWidget(self.newchat_btn)
        side_layout.addStretch(1)

        root_layout.addWidget(sidebar)

        # 历史抽屉
        self.history_drawer = QtWidgets.QFrame()
        self.history_drawer.setMaximumWidth(0)
        self.history_drawer.setMinimumWidth(0)
        self.history_drawer.setStyleSheet("QFrame{background:#FFFFFF; border-right:1px solid #E0E0E0;}")
        drawer_layout = QtWidgets.QVBoxLayout(self.history_drawer)
        drawer_layout.setContentsMargins(12, 12, 12, 12)
        drawer_layout.setSpacing(8)
        drawer_title = QtWidgets.QLabel("历史记录")
        drawer_title.setStyleSheet("font-weight:bold; color:#333;")
        self.history_list = QtWidgets.QListWidget()
        self.history_list.setStyleSheet(
            "QListWidget{border:none;}"
            "QListWidget::item{padding:8px; margin:6px; border:1px solid #E0E0E0; border-radius:8px;}"
            "QListWidget::item:selected{background:#EAF2FF; border:1px solid #3399FF;}"
        )
        drawer_layout.addWidget(drawer_title)
        drawer_layout.addWidget(self.history_list)
        root_layout.addWidget(self.history_drawer)
        self.history_list.itemClicked.connect(self.on_load_session)

        main = QtWidgets.QWidget()
        root_layout.addWidget(main)

        # 主布局
        layout = QtWidgets.QVBoxLayout(main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- 顶部区域 (可选，增加一点层次感) ---
        # 可以在这里加个标题栏，但为了保持简洁先略过

        # --- 中间聊天区域（气泡） ---
        self.chat_area = QtWidgets.QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setStyleSheet("QScrollArea{border:none; background:#F0F2F5;}")
        self.chat_container = QtWidgets.QWidget()
        self.chat_layout = QtWidgets.QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(16, 16, 16, 16)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch(1)
        self.chat_area.setWidget(self.chat_container)
        layout.addWidget(self.chat_area, 1)

        # --- 拖拽面板 (浮层或嵌入) ---
        self.drop_container = QtWidgets.QWidget()
        self.drop_container.setVisible(False)
        self.drop_container.setStyleSheet("background-color: #FFFFFF; border-bottom: 1px solid #E0E0E0;")
        drop_layout = QtWidgets.QHBoxLayout(self.drop_container)
        drop_layout.setContentsMargins(20, 10, 20, 10)
        drop_layout.setSpacing(20)
        
        # 配色微调：Note用蓝色，Paper用紫色，更现代
        self.note_panel = DropPanel("笔记 (Note)", "#3399FF", "note", self._add_files)
        self.paper_panel = DropPanel("论文 (Paper)", "#9B59B6", "paper", self._add_files)
        
        drop_layout.addWidget(self.note_panel)
        drop_layout.addWidget(self.paper_panel)
        layout.addWidget(self.drop_container)

        # --- 底部输入区域 ---
        bottom_area = QtWidgets.QWidget()
        bottom_area.setStyleSheet("background-color: #F7F7F7; border-top: 1px solid #E5E5E5;")
        bottom_layout = QtWidgets.QHBoxLayout(bottom_area)
        bottom_layout.setContentsMargins(20, 15, 20, 15)
        bottom_layout.setSpacing(12)

        # 1. “知”字按钮
        self.add_btn = QtWidgets.QPushButton("知")
        self.add_btn.setToolTip("知识库管理")
        self.add_btn.clicked.connect(self.on_toggle_panel)
        self.add_btn.setFixedSize(36, 36)
        # 使用衬线体（Times/Songti）增加“知识”的厚重感
        self.add_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333; 
                color: #F0F0F0; 
                border-radius: 18px; 
                font-family: "Times New Roman", "SimSun", serif; 
                font-size: 20px; 
                font-weight: bold;
                border: 2px solid #333;
            }
            QPushButton:hover {
                background-color: #555;
                border-color: #555;
            }
        """)
        
        # 2. 重建索引按钮
        self.reindex_btn = QtWidgets.QPushButton("↻")
        self.reindex_btn.setToolTip("刷新索引")
        self.reindex_btn.clicked.connect(self.on_reindex)
        self.reindex_btn.setFixedSize(36, 36)
        self.reindex_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #666;
                border: 1px solid #CCC;
                border-radius: 18px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #E0E0E0;
                color: #333;
            }
        """)

        # 3. 输入框
        self.entry = QtWidgets.QLineEdit()
        self.entry.setPlaceholderText("请输入您的问题...")
        self.entry.setMinimumHeight(40)
        self.entry.returnPressed.connect(self.on_send)

        # 4. 发送按钮
        send_btn = QtWidgets.QPushButton("发送")
        send_btn.setObjectName("sendBtn")
        send_btn.setMinimumHeight(40)
        send_btn.setCursor(QtCore.Qt.PointingHandCursor)
        send_btn.clicked.connect(self.on_send)

        bottom_layout.addWidget(self.add_btn)
        bottom_layout.addWidget(self.reindex_btn)
        bottom_layout.addWidget(self.entry)
        bottom_layout.addWidget(send_btn)
        
        layout.addWidget(bottom_area)

        # 状态栏 (浮动在右下角或者作为单独一行，这里保持简单)
        self.status = QtWidgets.QLabel("系统就绪")
        self.status.setObjectName("statusBar")
        self.status.setAlignment(QtCore.Qt.AlignRight)
        # 把它加到底部布局的最下面，或者作为 footer
        layout.addWidget(self.status)

        self._update_history_list()

    def _append_chat(self, role: str, text: str):
        if not self.loading_history:
            self.current_session.append({"role": role, "text": text})
            if self.loaded_from_history:
                self.session_dirty = True

        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 5, 0, 5) # 增加每条消息上下的间距
        row_layout.setSpacing(10) # 头像和气泡的间距

        is_user = role == "你"
        
        # 头像
        avatar = QtWidgets.QLabel()
        # 注意：这里稍微调小了头像，使其更精致
        avatar_pix = self._circle_avatar("我" if is_user else "AI", "#007AFF" if is_user else "#10A37F")
        avatar.setPixmap(avatar_pix)
        avatar.setFixedSize(40, 40)
        
        # 气泡 (不再需要外部设置宽度)
        bubble = BubbleWidget(text, is_user)

        if is_user:
            row_layout.addStretch(1)
            # AlignTop 让头像对齐气泡顶部
            row_layout.addWidget(bubble, 0, QtCore.Qt.AlignTop)
            row_layout.addWidget(avatar, 0, QtCore.Qt.AlignTop)
        else:
            row_layout.addWidget(avatar, 0, QtCore.Qt.AlignTop)
            row_layout.addWidget(bubble, 0, QtCore.Qt.AlignTop)
            row_layout.addStretch(1)

        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row)
        QtCore.QTimer.singleShot(50, self._scroll_to_bottom)
    
    def _circle_avatar(self, label: str, color: str) -> QtGui.QPixmap:
        size = 40 # 配合上面的 FixedSize
        pix = QtGui.QPixmap(size, size)
        pix.fill(QtCore.Qt.transparent)
        
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # 绘制圆形背景
        painter.setBrush(QtGui.QColor(color))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(0, 0, size, size)
        
        # 绘制文字
        painter.setPen(QtGui.QColor("#ffffff"))
        # 使用稍微小一点的字体，防止“AI”两个字撑满
        font = QtGui.QFont("Microsoft YaHei", 12, QtGui.QFont.Bold)
        painter.setFont(font)
        painter.drawText(pix.rect(), QtCore.Qt.AlignCenter, label)
        
        painter.end()
        return pix

    def _scroll_to_bottom(self):
        bar = self.chat_area.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_status(self, text: str):
        self.status.setText(text)

    # ... (_ensure_llm_retriever, on_send, _add_files, on_reindex 等逻辑保持不变) ...
    def _ensure_llm_retriever(self):
        if self.retriever is None:
            self.retriever = get_retriever(
                chroma_dir=self.paths["vector_store_dir"],
                model_name=self.settings.get("embedding.model_name"),
                device=self.settings.get("embedding.device"),
                batch_size=self.settings.get("embedding.batch_size"),
                top_k=self.settings.get("retriever.top_k"),
                chunks_file=self.paths.get("chunks_file", ""),
                top_k_vector=self.settings.get("retriever.top_k_vector"),
                top_k_bm25=self.settings.get("retriever.top_k_bm25"),
                top_k_final=self.settings.get("retriever.top_k_final"),
                rerank_model=self.settings.get("rerank.model_name") or "",
            )
        if self.llm is None:
            self.llm = ChatOpenAI(
                base_url=self.settings.get("llm.api_base"),
                api_key=self.settings.get("llm.api_key"),
                model=self.settings.get("llm.model"),
                temperature=self.settings.get("llm.temperature"),
            )
        if getattr(self, "summary_llm", None) is None:
            if self.settings.get("summary.enabled"):
                self.summary_llm = ChatOpenAI(
                    base_url=self.settings.get("llm.api_base"),
                    api_key=self.settings.get("llm.api_key"),
                    model=self.settings.get("summary.model") or self.settings.get("llm.model"),
                    temperature=self.settings.get("summary.temperature", 0.0),
                )
            else:
                self.summary_llm = None

    def on_send(self):
        question = self.entry.text().strip()
        if not question:
            return
        self.entry.clear()
        self._append_chat("你", question)
        self._set_status("正在思考...")

        def task():
            try:
                self._ensure_llm_retriever()
                history = self.current_session[-6:]
                answer = answer_question(
                    question,
                    self.retriever,
                    self.llm,
                    chat_history=history,
                    summary_llm=getattr(self, "summary_llm", None),
                    summary_cfg={
                        "max_chars_per_chunk": self.settings.get("summary.max_chars_per_chunk"),
                        "max_context_chars": self.settings.get("summary.max_context_chars"),
                    },
                )
                self.signals.append_chat.emit("助手", answer)
                self.signals.set_status.emit("就绪")
            except Exception as exc:
                self.signals.append_chat.emit("系统", f"发生错误：{exc}")
                self.signals.set_status.emit("出错")

        threading.Thread(target=task, daemon=True).start()

    def _add_files(self, category: str, files: Iterable[str]):
        try:
            saved = add_files_to_category(self.paths["knowledge_base_dir"], category, files)
            self._set_status(f"成功添加 {len(saved)} 个文件到 {category}")
            self.drop_container.setVisible(False)
        except Exception as exc:
            self._append_chat("系统", f"导入失败：{exc}")
            self._set_status("导入失败")

    def on_toggle_panel(self):
        self.drop_container.setVisible(not self.drop_container.isVisible())
        if self.drop_container.isVisible():
            self._set_status("请拖拽文件到上方区域")

    def on_reindex(self):
        self._set_status("正在重建索引...")
        def task():
            try:
                count = run_ingest(self.settings, self.paths)
                self.retriever = None 
                self.signals.set_status.emit(f"索引完成，当前向量数：{count}")
            except Exception as exc:
                self.signals.append_chat.emit("系统", f"索引失败：{exc}")
                self.signals.set_status.emit("索引失败")
        threading.Thread(target=task, daemon=True).start()

    def on_show_history(self):
        self._toggle_history_drawer()

    def on_new_chat(self):
        self._save_current_session()
        self.current_session = []
        self._clear_chat_view()
        self._set_status("已开始新对话")
        self.loaded_from_history = False
        self.session_dirty = False
        self.loaded_session_index = None

    def _update_history_list(self):
        self.history_list.clear()
        for idx, session in enumerate(self._load_history()):
            title = session.get("title") or f"对话 {idx + 1}"
            ts = session.get("time", "")
            label = f"{ts}  {title}" if ts else title
            self.history_list.addItem(label)

    def _toggle_history_drawer(self):
        start = self.history_drawer.maximumWidth()
        end = 260 if start == 0 else 0
        anim = QtCore.QPropertyAnimation(self.history_drawer, b"maximumWidth", self)
        anim.setDuration(220)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.start()
        self._history_anim = anim

    def closeEvent(self, event: QtGui.QCloseEvent):
        # 退出时保存当前对话（有内容才保存）
        self._save_current_session()
        super().closeEvent(event)

    def _clear_chat_view(self):
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    def _load_history(self):
        if not os.path.isfile(self.history_path):
            return []
        try:
            import json

            with open(self.history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_history(self, sessions):
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        import json

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)

    def _save_current_session(self):
        if not self.current_session:
            return
        sessions = self._load_history()
        title = self.current_session[0]["text"][:24] if self.current_session else "新对话"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        if self.loaded_from_history and self.loaded_session_index is not None:
            if not self.session_dirty:
                return
            if 0 <= self.loaded_session_index < len(sessions):
                sessions[self.loaded_session_index] = {
                    "title": title,
                    "time": ts,
                    "messages": self.current_session,
                }
            else:
                sessions.insert(0, {"title": title, "time": ts, "messages": self.current_session})
        else:
            sessions.insert(0, {"title": title, "time": ts, "messages": self.current_session})
        self._save_history(sessions[:50])
        self._update_history_list()

    def on_load_session(self, item):
        idx = self.history_list.row(item)
        # 切换历史前先保存当前对话（若有内容）
        self._save_current_session()
        sessions = self._load_history()
        if idx < 0 or idx >= len(sessions):
            return
        self.current_session = copy.deepcopy(sessions[idx].get("messages", []))
        self._clear_chat_view()
        self.loading_history = True
        for msg in self.current_session:
            self._append_chat(msg.get("role", ""), msg.get("text", ""))
        self.loading_history = False
        self.loaded_from_history = True
        self.session_dirty = False
        self.loaded_session_index = idx

def launch_ui(config_path: str):
    app = QtWidgets.QApplication([])
    font = QtGui.QFont("PingFang SC", 10)
    app.setFont(font)
    win = ChatWindow(config_path)
    app.aboutToQuit.connect(win._save_current_session)
    win.show()
    app.exec()
