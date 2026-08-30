# -*- coding: utf-8 -*-
"""
函数曲线绘图器（学习增强版）

一个带图形界面的函数绘图学习工具：
- 输入函数表达式即可画出曲线，支持多条曲线、x 轴范围设置；
- “变换曲线”：用中文自然语言描述几何变换（平移、伸缩、对称、旋转），
  优先由大语言模型解析，解析失败自动退回规则匹配；
- “修改函数”：用自然语言描述修改要求，由大语言模型生成新的函数表达式；
- 内置数值导数、切线、定积分面积等学习功能，帮助直观理解变化率与面积；
- 缩放/平移时曲线会按可见视野自动延伸重绘；常用按钮集中在顶部，
  “修改函数”为内置对话面板。

安全说明：表达式采用 AST 白名单解析，只允许注册的数学函数，
不使用裸 eval，避免恶意代码执行。
"""

import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog, scrolledtext
import re
import ast
import json
import os
import sys
import numpy as np
import matplotlib
import requests

matplotlib.use("TkAgg")  # 让 matplotlib 的图嵌入到 tkinter 窗口中

# 设置中文字体，避免图例里的中文显示成方块
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 让负号正常显示

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 可以安全使用的数学函数，表达式计算时会用到这些名字
SAFE_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,      # 自然对数 ln(x)
    "log10": np.log10,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
}

# DeepSeek 大模型相关配置
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
# PyInstaller 打包后：程序目录跟随 exe，图标从解压目录读取，config 保存在 exe 旁边
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
    ICON_DIR = os.path.join(getattr(sys, "_MEIPASS", BASE_DIR), "icons")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ICON_DIR = os.path.join(BASE_DIR, "icons")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")  # 保存 API Key 的本地文件
DEFAULT_X_RANGE = (-10.0, 10.0)  # 默认采样范围（不再需要手动设置）

# 对话中识别为“图像变换”的关键词，其余按“修改表达式”处理
_TRANSFORM_KEYWORDS = (
    "平移", "拉伸", "压缩", "对称", "翻折", "旋转", "放大", "缩小", "伸缩",
)

# 二次函数“开口”相关指令：本地规则优先，避免大模型理解方向反了
# （y=a*x**2 中 |a| 越大开口越窄，越小开口越宽）
_OPEN_NARROW_KEYS = ("收窄", "变窄", "开口缩小")
_OPEN_WIDE_KEYS = ("变宽", "开口放大", "开口扩大")



# ---------------------------------------------------------------
# 安全表达式求值：AST 白名单
# ---------------------------------------------------------------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.FloorDiv, ast.USub, ast.UAdd, ast.Call, ast.keyword,
)


def _safe_eval(expr, namespace):
    """用 AST 白名单方式安全地计算数学表达式。

    只允许四则运算、幂、一元正负号和已注册的数学函数调用，
    任何属性访问、下标、导入等操作都会被拒绝。
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误：{exc.msg}") from exc

    def _check(node):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError("表达式中包含不允许的语法元素")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in namespace:
                raise ValueError("只能调用白名单中的数学函数")
        for child in ast.iter_child_nodes(node):
            _check(child)

    _check(tree)
    # 命名空间里没有 __builtins__，即使 AST 允许的节点也无法触达系统能力
    # 数学函数在定义域外的取值（如 sqrt 负数、log 负数）只是 NaN，
    # 屏蔽 numpy 的告警，避免控制台出现“invalid value”之类的提示
    with np.errstate(all="ignore"):
        return eval(compile(tree, "<safe_expr>", "eval"), {"__builtins__": {}}, namespace)


# ---------------------------------------------------------------
# 表达式规范化：把数学课写法转成 Python 写法
# ---------------------------------------------------------------
# 每个函数名对应一个大写字母“占位符”，
# 先保护函数名再补乘号，避免误伤 sin、log10 等名字。
_FUNC_CODE = {
    "sin": "S", "cos": "C", "tan": "T", "asin": "A", "acos": "B",
    "atan": "D", "sqrt": "Q", "exp": "E", "log": "L", "log10": "K",
    "abs": "U", "pi": "P", "e": "M",
}
_CODE_TO_FUNC = {v: k for k, v in _FUNC_CODE.items()}
_FUNCTION_NAMES = sorted(_FUNC_CODE, key=len, reverse=True)  # log10 排在 log 前


def _normalize_expr(expr):
    """把数学课上的常见写法转成 Python 能识别的表达式。

    处理：去空格、^ -> **、补省略的乘号。
    支持 4x、0.5x、x^2、2sin(x)、x(x+1)、(x+1)(x-1)、2pi、pi x 等写法。
    """
    expr = re.sub(r"\s+", "", expr.lower())
    expr = expr.replace("^", "**")

    # 逐字符扫描，把函数名替换成大写占位符
    out = []
    i, n = 0, len(expr)
    prev_was_token = False  # 前一个字符是否来自函数名占位符
    while i < n:
        # 科学计数法里的 e（如 2e3）：用 § 占位，避免被当作常数 e
        if (
            expr[i] == "e"
            and (i == 0 or expr[i - 1] != "_")
            and i + 1 < n
            and expr[i + 1].isdigit()
        ):
            out.append("§")
            i += 1
            prev_was_token = False
            continue

        matched = None
        for name in _FUNCTION_NAMES:
            if not expr.startswith(name, i):
                continue
            j = i + len(name)
            # 前一个字符允许是：开头、数字、变量 x、上一个函数占位符
            prev_ok = (
                i == 0
                or prev_was_token
                or expr[i - 1].isdigit()
                or expr[i - 1] == "x"
            )
            # 后一个字符允许是：结尾、非字母数字，或变量 x（如 pi x）
            next_ok = (
                j >= n
                or not (expr[j].isalnum() or expr[j] == "_")
                or expr[j] == "x"
            )
            if prev_ok and next_ok:
                matched = name
                break
        if matched:
            out.append(_FUNC_CODE[matched])
            i += len(matched)
            prev_was_token = True
        else:
            out.append(expr[i])
            i += 1
            prev_was_token = False
    expr = "".join(out)

    # 补省略的乘号（此时小写字母只剩变量 x，大写字母都是函数占位符）
    expr = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", expr)      # 2x、3(x+1)、2sin(x)、2pi
    expr = re.sub(r"(\))([a-zA-Z0-9(])", r"\1*\2", expr)   # (x+1)(x-1)、(x+1)2、(x+1)sin(x)
    expr = re.sub(r"(x)([a-zA-Z(])", r"\1*\2", expr)       # x(x+1)、xsin(x)
    expr = re.sub(r"([A-Z])([A-Z])", r"\1*\2", expr)       # 函数名相邻：pi*sin(x)
    expr = re.sub(r"([A-Z])(x)", r"\1*\2", expr)           # 函数名后跟 x：pi*x

    # 还原函数名（§ 是科学计数法里的 e）
    expr = re.sub(r"[A-Z]", lambda m: _CODE_TO_FUNC[m.group()], expr)
    expr = expr.replace("§", "e")
    return expr


class ToolTip:
    """简单的悬停提示气泡"""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tip = None
        widget.bind("<Enter>", lambda e: self._show(), add="+")
        widget.bind("<Leave>", lambda e: self._hide(), add="+")
        widget.bind("<ButtonPress>", lambda e: self._hide(), add="+")

    def _show(self):
        if self._tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tip, text=self.text, bg="#ffffe8", relief="solid", bd=1,
            font=("Microsoft YaHei", 9), padx=6, pady=3,
        ).pack()

    def _hide(self):
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class FunctionPlotter:
    def __init__(self, root):
        self.root = root
        root.title("LYF 函数曲线绘图器 V1.0")
        root.geometry("1120x740")

        self.curves = []  # 存所有曲线：每个元素是 {"x": ..., "y": ..., "label": ...}
        self.api_key = None  # DeepSeek API Key，首次使用时读取或询问
        self._pending_expr = None  # 对话面板中等待应用的函数表达式
        self._syncing_view = False  # 防止视野变化事件递归重绘
        self._icons = {}  # 按钮图标引用，防止被垃圾回收
        self._last_xlim = None  # 上次绘制时的 x 轴范围（用于平移等场景的延伸兜底）

        self._build_ui()
        self._init_plot()

    def _build_ui(self):
        """创建顶部图标工具栏、典型函数、提示、对话面板与状态栏"""
        bg = "#edf1f5"

        # ---------- 最顶部：常规功能图标按钮 ----------
        bar0 = tk.Frame(self.root, bg=bg)
        bar0.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 0))

        # 平移/缩放是持续模式：按下凹陷、再点取消
        self._mode_buttons = {}
        self._mode_buttons["pan"] = self._tool_button(
            bar0, "pan", "平移", self._toggle_tool_mode("pan"),
            "按下后拖动画布平移视野；再点一下取消",
        )
        self._mode_buttons["zoom"] = self._tool_button(
            bar0, "zoom", "缩放", self._toggle_tool_mode("zoom"),
            "按下后按住左键框选放大；再点一下取消",
        )
        self._sep(bar0)

        for icon, text, method, tip in (
            ("back", "后退", "back", "回到上一个视图"),
            ("forward", "前进", "forward", "前进到下一个视图"),
            ("home", "复位", "home", "回到初始视图"),
            ("save", "保存", "save_figure", "把当前图像保存为 PNG"),
        ):
            self._tool_button(
                bar0, icon, text,
                lambda m=method: getattr(self.toolbar, m)(), tip,
            )
        self._sep(bar0)
        self._tool_button(bar0, "clear", "清除", self.clear_all, "清除全部曲线")

        # ---------- 第二行：函数输入 + 典型函数一键填入 ----------
        bar1 = tk.Frame(self.root, bg=bg)
        bar1.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(2, 4))

        tk.Label(bar1, text="函数 y =", bg=bg, font=("Microsoft YaHei", 10)).pack(
            side=tk.LEFT, padx=(4, 2)
        )
        self.expr_var = tk.StringVar()
        self.entry = tk.Entry(
            bar1, textvariable=self.expr_var, width=42, font=("Consolas", 12)
        )
        self.entry.pack(side=tk.LEFT, padx=4)
        self.entry.bind("<Return>", lambda e: self.add_function())  # 回车也能添加

        tk.Button(
            bar1, text="添加曲线", command=self.add_function,
            bd=1, relief=tk.RAISED, font=("Microsoft YaHei", 9),
            bg="#ffffff", activebackground="#dde5ee", padx=6, pady=2,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            bar1, text="清空", command=self.clear_input,
            bd=1, relief=tk.RAISED, font=("Microsoft YaHei", 9),
            bg="#ffffff", activebackground="#dde5ee", padx=6, pady=2,
        ).pack(side=tk.LEFT, padx=2)

        tk.Label(bar1, text="典型函数:", bg=bg, fg="#666666",
                 font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=(10, 2))
        for expr in ("x^2", "x^3", "sin(x)", "cos(x)", "1/x", "sqrt(x)", "log(x)"):
            lab = tk.Label(
                bar1, text=expr, bg=bg, fg="#1a6fce", cursor="hand2",
                font=("Consolas", 9, "underline"),
            )
            lab.pack(side=tk.LEFT, padx=4)
            lab.bind("<Button-1>", lambda e, ex=expr: self._fill_expr(ex))

        # ---------- 提示条 ----------
        hint = tk.Label(
            self.root,
            text="提示：支持 sin/cos/tan/asin/acos/atan/sqrt/exp/log/log10/abs/pi/e；"
            "可写 4x、x^2、x(x+1)、2sin(x)。缩放/平移时曲线自动延伸重绘；"
            "求导/切线/定积分/变换等请点下方“常用指令”链接，或直接在对话输入。",
            fg="#666666",
            bg="#f5f7fa",
            anchor="w",
            padx=10,
            pady=3,
        )
        hint.pack(side=tk.TOP, fill=tk.X)

        # ---------- 状态栏 ----------
        self.status_var = tk.StringVar(value="就绪：输入函数表达式后按回车")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            fg="#555555",
            bg="#f5f7fa",
            anchor="w",
            padx=10,
            pady=1,
        ).pack(side=tk.BOTTOM, fill=tk.X)

        # ---------- AI 对话面板（变换 / 修改函数） ----------
        chat_frame = tk.Frame(self.root)
        chat_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(4, 2))

        tk.Label(
            chat_frame,
            text="AI 助手对话（输入指令即可变换或修改函数）：",
            fg="#333333",
            anchor="w",
        ).pack(side=tk.TOP, fill=tk.X)

        quick = tk.Frame(chat_frame, bg="#f2f5f9")
        quick.pack(side=tk.TOP, fill=tk.X, pady=(2, 4))
        tk.Label(quick, text="常用指令:", bg="#f2f5f9", fg="#666666",
                 font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=(4, 4))
        quick_cmds = (
            ("平移", lambda: self._ask_transform("平移", "向上平移2个单位")),
            ("旋转", lambda: self._ask_transform("旋转", "绕原点旋转90度")),
            ("对称", lambda: self._ask_transform("对称", "关于x轴对称")),
            ("拉伸", lambda: self._ask_transform("拉伸", "纵向拉伸2倍")),
            ("求导", self.add_derivative),
            ("切线", self.add_tangent),
            ("定积分", self.add_integral),
            ("改系数", self._ask_modify),
        )
        for i, (text, cmd) in enumerate(quick_cmds):
            if i:
                tk.Label(quick, text="|", bg="#f2f5f9", fg="#b8c0ca").pack(side=tk.LEFT)
            lab = tk.Label(
                quick, text=text, bg="#f2f5f9", fg="#1a6fce", cursor="hand2",
                font=("Microsoft YaHei", 9, "underline"),
            )
            lab.pack(side=tk.LEFT, padx=4, pady=2)
            lab.bind("<Button-1>", lambda e, c=cmd: c())

        self.chat_text = scrolledtext.ScrolledText(
            chat_frame,
            height=5,
            state="disabled",
            wrap="word",
            font=("Microsoft YaHei", 9),
        )
        self.chat_text.pack(side=tk.TOP, fill=tk.X)

        chat_input = tk.Frame(chat_frame)
        chat_input.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        self.chat_var = tk.StringVar()
        self.chat_entry = tk.Entry(chat_input, textvariable=self.chat_var)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chat_entry.bind("<Return>", lambda e: self.chat_send())
        tk.Button(chat_input, text="发送", command=self.chat_send).pack(
            side=tk.LEFT, padx=4
        )
        self.apply_btn = tk.Button(
            chat_input, text="应用到曲线", command=self.chat_apply, state="disabled"
        )
        self.apply_btn.pack(side=tk.LEFT, padx=4)

        self._chat_append("系统", "欢迎使用！先画一条函数曲线，再在下方输入指令："
                                  "平移/拉伸/对称等变换，或“把常数项加2”等修改，也可直接点常用指令。")

    def _tool_button(self, parent, icon, text, command, tip=""):
        """创建带图标的顶部按钮（图标在上、文字在下，悬停显示提示）"""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            compound=tk.TOP,
            bd=1,
            relief=tk.RAISED,
            font=("Microsoft YaHei", 8),
            bg="#f8f9fa",
            activebackground="#dde5ee",
            padx=4,
            pady=2,
            cursor="hand2",
        )
        img = self._load_icon(icon)
        if img is not None:
            btn.config(image=img)
        btn.pack(side=tk.LEFT, padx=2, pady=3)
        if tip:
            ToolTip(btn, tip)
        return btn

    def _load_icon(self, name):
        """读取图标 PNG 并返回 tk.PhotoImage（保留引用防止被回收）"""
        path = os.path.join(ICON_DIR, name + ".png")
        if not os.path.exists(path):
            return None
        img = None
        try:
            img = tk.PhotoImage(file=path)
        except Exception:
            img = None
        if img is None:  # Tk 不支持该 PNG 时，用 PIL 转换后显示
            try:
                from PIL import Image, ImageTk

                img = ImageTk.PhotoImage(Image.open(path))
            except Exception:
                return None
        self._icons[name] = img
        return img

    @staticmethod
    def _sep(parent):
        """工具栏中的竖向分隔线"""
        s = tk.Frame(parent, bg="#c8cfd8", width=2)
        s.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=4)
        return s

    def _toggle_tool_mode(self, mode):
        """返回平移/缩放的开关函数：按下凹陷、再点取消"""
        def _toggle():
            if mode == "pan":
                self.toolbar.pan()
            else:
                self.toolbar.zoom()
            self._sync_mode_buttons()
        return _toggle

    def _sync_mode_buttons(self):
        """根据工具栏当前模式刷新平移/缩放按钮的凹陷效果"""
        toolbar_mode = getattr(self.toolbar, "mode", "")
        states = {
            "pan": toolbar_mode == "pan/zoom",
            "zoom": toolbar_mode == "zoom rect",
        }
        for name, btn in self._mode_buttons.items():
            if states.get(name):
                btn.config(relief=tk.SUNKEN, bg="#cfe0f2")
            else:
                btn.config(relief=tk.RAISED, bg="#f8f9fa")

    def _init_plot(self):
        """初始化画布和坐标轴"""
        self.fig = Figure(figsize=(9, 5.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._reset_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)  # 滚轮缩放
        # 平移时 matplotlib 不触发 xlim_changed，用绘制完成事件兜底比对视野
        self.canvas.mpl_connect("draw_event", self._on_draw_event)
        # 视野（x 轴范围）变化时延伸曲线；这是坐标轴的回调，不是画布事件
        self.ax.callbacks.connect("xlim_changed", self._on_view_changed)
        # 工具栏按钮改为顶部快捷按钮，这里只借用其功能、不显示底栏
        try:
            self.toolbar = NavigationToolbar2Tk(
                self.canvas, self.root, pack_toolbar=False
            )
        except TypeError:  # 兼容旧版 matplotlib
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
            self.toolbar.pack_forget()

    def _on_scroll(self, event):
        """鼠标滚轮以光标为中心缩放"""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        nxmin = event.xdata - (event.xdata - xmin) * factor
        nxmax = event.xdata + (xmax - event.xdata) * factor
        nymin = event.ydata - (event.ydata - ymin) * factor
        nymax = event.ydata + (ymax - event.ydata) * factor

        # 先抑制事件，避免中间重绘；改完视野后一次性延伸重绘曲线
        self._syncing_view = True
        try:
            self.ax.set_xlim(nxmin, nxmax)
            self.ax.set_ylim(nymin, nymax)
        finally:
            self._syncing_view = False
        self._redraw(auto_zoom=False, recompute=True)

    def _on_view_changed(self, event):
        """x 轴视野变化（缩放/平移/复位）后，按新视野重算并延伸曲线"""
        if self._syncing_view:
            return
        self._syncing_view = True
        try:
            self._redraw(auto_zoom=False, recompute=True)
        except Exception:
            pass
        finally:
            self._syncing_view = False

    def _reset_axes(self):
        """画好坐标轴、网格等背景"""
        self.ax.clear()
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.axhline(0, color="black", linewidth=0.8)  # x 轴
        self.ax.axvline(0, color="black", linewidth=0.8)  # y 轴
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

    def _get_x_range(self):
        """返回默认采样范围（x 范围已改为自动，不再需要手动设置）"""
        return DEFAULT_X_RANGE

    @staticmethod
    def _to_y_array(y, x):
        """把计算结果整理成一维数组；结果无效时返回 None"""
        try:
            y = np.asarray(y, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if y.size != x.size:
            return None
        # 把无效值（如 log 的负数、tan 的间断点）设为 NaN，避免画出错误连线
        y = np.where(np.isfinite(y), y, np.nan)
        return FunctionPlotter._break_poles(y)

    @staticmethod
    def _break_poles(y):
        """在疑似垂直渐近线处断开连线。

        例如 1/x 在 x=0 附近：相邻采样点符号相反且数值都远大于整体中位数，
        说明中间穿过了一条垂直渐近线，把边界点设为 NaN 即可断开竖线。
        """
        y = np.array(y, dtype=float, copy=True)
        if y.size < 4:
            return y
        finite = np.abs(y[np.isfinite(y)])
        if finite.size < 4:
            return y
        med = np.median(finite)
        if not np.isfinite(med) or med <= 0:
            return y
        threshold = 10.0 * med
        flip = np.signbit(y[:-1]) != np.signbit(y[1:])
        huge = (np.abs(y[:-1]) > threshold) & (np.abs(y[1:]) > threshold)
        idx = np.flatnonzero(flip & huge)
        y[idx] = np.nan
        return y

    def _last_expr_curve(self):
        """返回最后一条由表达式生成的曲线（导数/切线等派生曲线跳过）"""
        for c in reversed(self.curves):
            if c.get("expr"):
                return c
        return None

    def _recompute_curves(self, xmin, xmax, n=1200):
        """按给定 x 范围重算所有曲线的数据（缩放/平移时用于延伸绘制）"""
        x = np.linspace(xmin, xmax, n)
        ns = {**SAFE_FUNCS, "x": x}
        nan = np.full_like(x, np.nan)
        for c in self.curves:
            kind = c.get("kind", "function")
            try:
                if kind == "function":
                    y = self._to_y_array(_safe_eval(c["expr"], ns), x)
                    c["x"], c["y"] = x, y if y is not None else nan
                elif kind == "derivative":
                    y = self._to_y_array(_safe_eval(c["source_expr"], ns), x)
                    ys = y if y is not None else nan
                    c["x"], c["y"] = x, np.gradient(ys, x)
                elif kind == "tangent":
                    y = self._to_y_array(_safe_eval(c["source_expr"], ns), x)
                    if y is None:
                        c["x"], c["y"] = x, nan
                        continue
                    idx = int(np.argmin(np.abs(x - c["x0"])))
                    xp, yp = x[idx], y[idx]
                    slope = np.gradient(y, x)[idx]
                    c["x0"], c["y0"] = xp, yp
                    c["x"], c["y"] = x, yp + slope * (x - xp)
                elif kind == "area":
                    y = self._to_y_array(_safe_eval(c["source_expr"], ns), x)
                    c["x"], c["y"] = x, y if y is not None else nan
                    lo, hi = c["a"], c["b"]
                    c["mask"] = (x >= lo) & (x <= hi)
                elif kind == "transformed":
                    y = self._to_y_array(_safe_eval(c["source_expr"], ns), x)
                    if y is None:
                        c["x"], c["y"] = x, nan
                        continue
                    nx, ny = self._apply_transform(x, y, c["tkind"], c["tparams"])
                    c["x"], c["y"] = nx, ny
            except Exception:
                c["x"], c["y"] = x, nan
        return x

    def add_function(self):
        """读取表达式并画出这条曲线"""
        expr = self.expr_var.get().strip()
        if not expr:
            return

        x_range = self._get_x_range()
        if x_range is None:
            return
        xmin, xmax = x_range

        # 自动把数学课写法（如 4x、x^2、x(x+1)）转成 Python 写法
        normalized = _normalize_expr(expr)

        # 用 AST 白名单安全计算表达式，只允许数学函数
        x = np.linspace(xmin, xmax, 1200)
        try:
            y = _safe_eval(normalized, {**SAFE_FUNCS, "x": x})
        except Exception as e:
            messagebox.showerror("表达式错误", f"无法计算 “{expr}”：\n{e}")
            return

        y = self._to_y_array(y, x)
        if y is None:
            messagebox.showerror("错误", "表达式结果必须是一个以 x 为变量的函数")
            return

        self.curves.append({
            "kind": "function",
            "expr": normalized,
            "label": expr,
            "linestyle": "-",
        })
        self._recompute_curves(xmin, xmax)
        self._redraw(auto_zoom=True)
        self.expr_var.set("")  # 清空输入框，方便输入下一条

    def _redraw(self, auto_zoom=True, recompute=False):
        """清空坐标轴重新画出所有曲线。

        auto_zoom=True 时按数据自动调整显示范围；
        recompute=True 时先按当前视野重算曲线（用于缩放延伸）。
        """
        if recompute:
            try:
                self._recompute_curves(*self.ax.get_xlim())
            except Exception:
                pass

        saved_xlim = saved_ylim = None
        if not auto_zoom:
            saved_xlim = self.ax.get_xlim()
            saved_ylim = self.ax.get_ylim()

        self._reset_axes()
        if saved_xlim is not None:
            self.ax.set_xlim(saved_xlim)
            self.ax.set_ylim(saved_ylim)

        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        for i, c in enumerate(self.curves):
            color = colors[i % len(colors)]
            if c.get("kind") == "area":
                self.ax.fill_between(
                    c["x"], c["y"], where=c.get("mask"),
                    alpha=0.25, color=color, label=c["label"],
                )
                self.ax.axvline(c["a"], color=color, linestyle=":", linewidth=1.2)
                self.ax.axvline(c["b"], color=color, linestyle=":", linewidth=1.2)
            else:
                self.ax.plot(
                    c["x"], c["y"], label=c["label"],
                    color=color, linestyle=c.get("linestyle", "-"),
                )
                if c.get("kind") == "tangent":
                    self.ax.scatter([c["x0"]], [c["y0"]], color=color, zorder=5, s=28)
        if self.curves:
            self.ax.legend(loc="best", fontsize=9)
            if auto_zoom:
                self._syncing_view = True
                try:
                    self._auto_zoom()
                finally:
                    self._syncing_view = False
        self.canvas.draw()
        self._last_xlim = self.ax.get_xlim()
        self._update_status()

    def _on_draw_event(self, event):
        """每次绘制完成后检查视野是否被外部改变（如平移），改变则延伸曲线"""
        if self._syncing_view:
            return
        try:
            xlim = self.ax.get_xlim()
        except Exception:
            return
        last = self._last_xlim
        if last is None:
            self._last_xlim = xlim
            return
        if abs(xlim[0] - last[0]) > 1e-9 or abs(xlim[1] - last[1]) > 1e-9:
            self._last_xlim = xlim
            self._syncing_view = True
            try:
                self._redraw(auto_zoom=False, recompute=True)
            except Exception:
                pass
            finally:
                self._syncing_view = False

    def _update_status(self):
        """更新底部状态栏"""
        n = len(self.curves)
        self._sync_mode_buttons()
        if n == 0:
            self.status_var.set("就绪：输入函数表达式后按回车")
        else:
            self.status_var.set(
                f"当前 {n} 条曲线 · 缩放/平移时曲线自动延伸重绘"
            )

    def _auto_zoom(self):
        """根据所有曲线的数据自动设置坐标轴范围，并留一点边距"""
        xs = np.concatenate([c["x"] for c in self.curves])
        ys = np.concatenate([c["y"] for c in self.curves])
        xs = xs[np.isfinite(xs)]
        ys = ys[np.isfinite(ys)]

        if xs.size:
            xmin, xmax = xs.min(), xs.max()
            if xmin == xmax:  # 防止范围相同导致报错
                xmin, xmax = xmin - 1, xmax + 1
            self.ax.set_xlim(xmin, xmax)

        if ys.size:
            ymin, ymax = ys.min(), ys.max()
            if ymin == ymax:
                ymin, ymax = ymin - 1, ymax + 1
            pad = (ymax - ymin) * 0.05 or 1.0
            self.ax.set_ylim(ymin - pad, ymax + pad)

    def add_derivative(self):
        """绘制最后一条函数的数值导数曲线"""
        target = self._last_expr_curve()
        if target is None:
            messagebox.showinfo("提示", "还没有函数曲线，请先画一条")
            return
        self.curves.append({
            "kind": "derivative",
            "source_expr": target["expr"],
            "label": f"f'(x)（{target['label']} 的数值导数）",
            "linestyle": "--",
        })
        self._redraw(recompute=True, auto_zoom=True)

    def add_tangent(self):
        """在指定 x 处画最后一条函数的切线，并标出切点"""
        target = self._last_expr_curve()
        if target is None:
            messagebox.showinfo("提示", "还没有函数曲线，请先画一条")
            return

        ans = simpledialog.askstring(
            "画切线",
            f"在 x = ? 处画 {target['label']} 的切线（如 1.5）：",
            parent=self.root,
        )
        if not ans:
            return
        try:
            x0 = float(ans.strip())
        except ValueError:
            messagebox.showerror("输入错误", "x 坐标必须是数字")
            return

        x, y = target["x"], target["y"]
        idx = int(np.argmin(np.abs(x - x0)))
        xp, yp = x[idx], y[idx]
        slope = np.gradient(y, x)[idx]
        if not np.isfinite(slope):
            messagebox.showerror("无法画切线", "该点处函数不可导（斜率不存在）")
            return

        self.curves.append({
            "kind": "tangent",
            "source_expr": target["expr"],
            "x0": xp,
            "label": f"切线 x={xp:g}（斜率 {slope:.3f}）",
            "linestyle": "--",
        })
        self._redraw(recompute=True, auto_zoom=True)

    def add_integral(self):
        """计算最后一条函数在 [a, b] 上的定积分，并涂色显示面积"""
        target = self._last_expr_curve()
        if target is None:
            messagebox.showinfo("提示", "还没有函数曲线，请先画一条")
            return

        ans = simpledialog.askstring(
            "定积分（面积）",
            f"对 {target['label']} 求定积分，输入区间 [a, b]，如：0, 3.14",
            parent=self.root,
        )
        if not ans:
            return
        try:
            a_s, b_s = ans.replace("，", ",").split(",")
            a, b = float(a_s), float(b_s)
        except ValueError:
            messagebox.showerror("输入错误", "请按“0,3.14”的格式输入两个数字")
            return

        lo, hi = min(a, b), max(a, b)
        # 在积分区间上加密采样，提高数值积分精度
        xd = np.linspace(lo, hi, 2001)
        try:
            yd = self._to_y_array(
                _safe_eval(target["expr"], {**SAFE_FUNCS, "x": xd}), xd
            )
        except Exception:
            yd = None

        x, y = target["x"], target["y"]
        if yd is None:
            # 表达式不可重算时退回使用原有采样点
            mask = (x >= lo) & (x <= hi)
            if not mask.any():
                messagebox.showerror("区间错误", "区间不在当前绘图范围内")
                return
            xs, ys = x[mask], y[mask]
        else:
            xs, ys = xd, yd

        finite = np.isfinite(ys)
        if finite.sum() < 2:
            messagebox.showerror("无法计算", "区间内函数值无效，无法计算面积")
            return

        try:
            value = np.trapezoid(ys[finite], xs[finite])  # 数值积分（梯形法）
        except AttributeError:
            value = np.trapz(ys[finite], xs[finite])       # 兼容旧版 numpy

        self.curves.append({
            "kind": "area",
            "source_expr": target["expr"],
            "a": lo,
            "b": hi,
            "label": f"定积分面积 ≈ {value:.4f}（[{lo:g}, {hi:g}]）",
            "linestyle": "-",
        })
        self._redraw(recompute=True, auto_zoom=True)

    def _get_api_key(self):
        """从本地 config.json 读取 API Key，没有则弹窗询问并保存"""
        if self.api_key:
            return self.api_key

        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                self.api_key = json.load(f).get("api_key", "")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self.api_key = ""

        if not self.api_key:
            key = simpledialog.askstring(
                "配置 DeepSeek API Key",
                "请输入你的 DeepSeek API Key（sk- 开头）：\n\n"
                "会保存在本地 config.json 中，下次无需重复输入。\n"
                "注意：config.json 含密钥，请勿分享或上传到公开仓库。",
                parent=self.root,
            )
            if key:
                self.api_key = key.strip()
                try:
                    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                        json.dump({"api_key": self.api_key}, f)
                except OSError:
                    pass
        return self.api_key

    def _parse_transform_ai(self, desc):
        """调用 DeepSeek 大模型把自然语言描述解析成 (kind, params)，失败返回 None"""
        api_key = self._get_api_key()
        if not api_key:
            return None

        prompt = (
            "你是一个函数图像变换解析器。用户会用中文描述对函数图像的变换，"
            "请把描述解析成 JSON。\n\n"
            "JSON 格式：{\"kind\": \"类型\", \"params\": {参数}}\n\n"
            "kind 只能是下面这些：\n"
            "- \"right\" 向右平移，params 含 {\"d\": 距离}（默认 1）\n"
            "- \"left\" 向左平移，params 含 {\"d\": 距离}（默认 1）\n"
            "- \"up\" 向上平移，params 含 {\"d\": 距离}（默认 1）\n"
            "- \"down\" 向下平移，params 含 {\"d\": 距离}（默认 1）\n"
            "- \"scale_x\" 横向伸缩，params 含 {\"k\": 倍数}（默认 2）\n"
            "- \"scale_y\" 纵向伸缩，params 含 {\"k\": 倍数}（默认 2）\n"
            "- \"flip_x\" 关于x轴对称，params 为 {}\n"
            "- \"flip_y\" 关于y轴对称，params 为 {}\n"
            "- \"rotate\" 绕原点旋转，params 含 {\"angle\": 角度，逆时针为正}\n\n"
            "规则：压缩/缩小用小于 1 的倍数（如横向压缩2倍是 scale_x 且 k=0.5）；"
            "关于原点对称等价于 rotate 且 angle=180。\n\n"
            f"用户描述：{desc}\n\n"
            "只输出 JSON，不要输出任何其他文字。"
        )

        valid = {"right", "left", "up", "down", "scale_x", "scale_y",
                 "flip_x", "flip_y", "rotate"}
        for _ in range(2):  # 失败重试一次
            try:
                resp = requests.post(
                    DEEPSEEK_API_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": DEEPSEEK_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        "temperature": 0,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                content = re.sub(r"^```[a-zA-Z]*\s*", "", content).strip()
                content = re.sub(r"\s*```$", "", content)
                data = json.loads(content)
            except Exception:
                continue  # 网络或解析出错，交给规则匹配兜底

            kind = data.get("kind")
            params = data.get("params", {}) or {}
            if kind not in valid:
                continue

            # 把参数统一转成数字，避免大模型返回字符串导致后续出错
            if kind in ("right", "left", "up", "down"):
                try:
                    params["d"] = float(params.get("d", 1))
                except (TypeError, ValueError):
                    params["d"] = 1.0
            elif kind in ("scale_x", "scale_y"):
                try:
                    params["k"] = float(params.get("k", 2))
                except (TypeError, ValueError):
                    params["k"] = 2.0
            elif kind == "rotate":
                try:
                    params["angle"] = float(params.get("angle", 90))
                except (TypeError, ValueError):
                    params["angle"] = 90.0
            return kind, params
        return None

    def _chat_append(self, who, text):
        """向对话面板追加一条消息"""
        self.chat_text.configure(state="normal")
        self.chat_text.insert(tk.END, f"{who}：{text}\n")
        self.chat_text.see(tk.END)
        self.chat_text.configure(state="disabled")

    def chat_send(self):
        """把指令发送给 AI：变换类指令直接执行，修改类指令生成新函数"""
        request = self.chat_var.get().strip()
        if not request:
            return
        target = self._last_expr_curve()
        if target is None:
            self._chat_append("系统", "请先画一条函数曲线，再发送指令")
            return

        self.chat_var.set("")
        self._chat_append("我", request)

        # 二次函数“开口”类指令：本地规则直接生成，保证方向正确
        if "开口" in request:
            new_expr = self._gen_opening_expr(target["expr"], request)
            if new_expr is not None and self._validate_expr(new_expr):
                self._finish_modify(new_expr)
                return
            self._chat_append("系统", "正在调用大模型处理，请稍候……")
            self.root.update_idletasks()

        # 变换类指令：平移/拉伸/对称/旋转等，走“解析参数 → 几何变换”
        if any(k in request for k in _TRANSFORM_KEYWORDS):
            self._chat_append("系统", f"正在变换 y = {target['expr']}……")
            parsed = self._parse_transform_ai(request)
            if parsed is None:
                parsed = self._parse_transform(request)  # 离线兜底
            if parsed is None:
                self._chat_append(
                    "系统",
                    "没理解这条变换指令，试试“向上平移2个单位”“纵向拉伸2倍”“关于x轴对称”",
                )
                return
            kind, params = parsed
            new_expr = self._transform_expr(target["expr"], kind, params)
            if new_expr:
                label = f"y = {new_expr}"
            else:
                label = f"{target['label']} → {request}"
            self.curves.append({
                "kind": "transformed",
                "source_expr": target["expr"],
                "tkind": kind,
                "tparams": params,
                "label": label,
                "linestyle": "--",
            })
            self._redraw(recompute=True, auto_zoom=True)
            self._chat_append("AI", f"已完成变换（虚线为新曲线）：{request}")
            if new_expr:
                self._chat_append("AI", f"新表达式：y = {new_expr}")
            else:
                self._chat_append("系统", "旋转后的曲线一般不是函数，无法写出新表达式")
            return

        # 修改类指令：让大模型生成新的函数表达式
        self._chat_append("系统", f"正在修改 y = {target['expr']}，请稍候……")
        self.root.update_idletasks()
        new_expr = self._gen_expr_ai(target["expr"], request)
        if new_expr is None:
            self._chat_append("系统", "生成失败：请检查网络或 API Key 后重试")
            return
        self._finish_modify(new_expr)

    def _validate_expr(self, expr):
        """本地试算表达式是否有效"""
        normalized = _normalize_expr(expr)
        probe = np.linspace(*self.ax.get_xlim(), 200)
        try:
            y = self._to_y_array(_safe_eval(normalized, {**SAFE_FUNCS, "x": probe}), probe)
        except Exception:
            return False
        return y is not None

    def _finish_modify(self, new_expr):
        """把生成的新表达式放入待应用状态并提示用户"""
        self._pending_expr = new_expr
        self.apply_btn.configure(state="normal")
        self._chat_append("AI", f"新函数：y = {new_expr}")
        self._chat_append("系统", "点击右侧“应用到曲线”即可绘制（虚线为修改后的函数）")

    def _gen_opening_expr(self, current_expr, request):
        """针对二次函数“开口”指令：收窄→系数乘以大于 1 的数，变宽→乘以小于 1 的数"""
        narrow = any(k in request for k in _OPEN_NARROW_KEYS)
        wide = any(k in request for k in _OPEN_WIDE_KEYS)
        if narrow == wide:
            return None
        factor = 2.0 if narrow else 0.5
        m = re.search(r"\d+(?:\.\d+)?", request)
        if m:
            try:
                num = float(m.group())
            except ValueError:
                num = None
            if num and num > 0:
                factor = num if narrow else 1.0 / num
        return FunctionPlotter._scale_quad_coef(current_expr, factor)

    @staticmethod
    def _scale_quad_coef(expr, factor):
        """把表达式中 x**2 项的系数乘以 factor（找不到二次项返回 None）"""
        m = re.search(r"([+-]?\d*\.?\d*)\s*\*\s*x\*\*2", expr)
        if m:
            coef_str = m.group(1)
            if coef_str in ("", "+"):
                coef = 1.0
            elif coef_str == "-":
                coef = -1.0
            else:
                try:
                    coef = float(coef_str)
                except ValueError:
                    return None
            new_coef = coef * factor
            head = expr[: m.start()]
            tail = expr[m.end():]
            return f"{head}{FunctionPlotter._fmt_coef(new_coef)}*x**2{tail}"
        m = re.search(r"(?<![0-9a-zA-Z_)])x\*\*2", expr)
        if m:
            head = expr[: m.start()]
            tail = expr[m.end():]
            return f"{head}{FunctionPlotter._fmt_coef(factor)}*x**2{tail}"
        return None

    @staticmethod
    def _fmt_coef(c):
        return f"{c:.6g}"

    @staticmethod
    def _transform_expr(expr, kind, params):
        """根据几何变换参数生成变换后的函数表达式；旋转返回 None"""
        def sub_x(e, repl):
            return re.sub(r"(?<![A-Za-z0-9_])x(?![A-Za-z0-9_])", repl, e)
        d = params.get("d")
        k = params.get("k")
        if kind == "right" and d is not None:
            return sub_x(expr, f"(x-{d:g})")
        if kind == "left" and d is not None:
            return sub_x(expr, f"(x+{d:g})")
        if kind == "up" and d is not None:
            return f"({expr})+{d:g}"
        if kind == "down" and d is not None:
            return f"({expr})-{d:g}"
        if kind == "scale_x" and k is not None:
            return sub_x(expr, f"(x/{k:g})")
        if kind == "scale_y" and k is not None:
            return f"{k:g}*({expr})"
        if kind == "flip_x":
            return f"-({expr})"
        if kind == "flip_y":
            return sub_x(expr, "(-x)")
        return None

    def _ask_transform(self, kind, example):
        """点击“平移/旋转/对称/拉伸”链接：先确认方向、单位等细节再执行"""
        prompts = {
            "平移": "输入平移方向和距离，例如：\n  向上平移2个单位\n  向左3个单位",
            "旋转": "输入旋转方式和角度，例如：\n  绕原点旋转90度（逆时针为正，可输入负数）",
            "对称": "输入对称方式，例如：\n  关于x轴对称\n  关于y轴对称\n  关于原点对称",
            "拉伸": "输入伸缩方式和倍数，例如：\n  纵向拉伸2倍\n  横向压缩3倍",
        }
        desc = simpledialog.askstring(f"{kind}参数", prompts[kind], parent=self.root)
        if not desc:
            return
        self.chat_var.set(desc.strip())
        self.chat_send()

    def _ask_modify(self):
        """点击“改系数”链接：先确认要修改的内容再执行"""
        desc = simpledialog.askstring(
            "修改函数",
            "输入修改要求，例如：\n  把二次项系数改成3\n  常数项加2\n  整体平方",
            parent=self.root,
        )
        if not desc:
            return
        self.chat_var.set(desc.strip())
        self.chat_send()

    def _fill_expr(self, expr):
        """点击典型函数：填入函数输入框"""
        self.expr_var.set(expr)
        self.entry.focus_set()

    def clear_input(self):
        """清空函数输入框"""
        self.expr_var.set("")
        self.entry.focus_set()

    def chat_apply(self):
        """把对话中生成的新函数应用到绘图"""
        if not self._pending_expr:
            return
        target = self._last_expr_curve()
        if target is None:
            self._chat_append("系统", "没有可修改的函数曲线")
            return

        new_expr = self._pending_expr
        normalized = _normalize_expr(new_expr)
        probe = np.linspace(*self.ax.get_xlim(), 200)
        try:
            y = self._to_y_array(
                _safe_eval(normalized, {**SAFE_FUNCS, "x": probe}), probe
            )
        except Exception as e:
            self._chat_append("系统", f"表达式无法计算：{e}")
            return
        if y is None:
            self._chat_append("系统", "生成的结果不是有效的函数表达式")
            return

        self.curves.append({
            "kind": "function",
            "expr": normalized,
            "label": f"y = {new_expr}",
            "linestyle": "--",
        })
        self._pending_expr = None
        self.apply_btn.configure(state="disabled")
        self._redraw(recompute=True, auto_zoom=True)
        self._chat_append("系统", f"已应用：y = {new_expr}")

    def _gen_expr_ai(self, current_expr, instruction):
        """调用大模型生成新表达式，并在本地验证可计算，失败返回 None"""
        api_key = self._get_api_key()
        if not api_key:
            return None

        prompt = (
            "你是一个数学函数编辑器。给出一个函数表达式和一句修改要求，"
            "请输出修改后的新函数表达式。\n\n"
            "要求：\n"
            "1. 使用 Python 能直接计算的写法（sin、cos、tan、sqrt、exp、log、"
            "abs 等；幂用 **，乘号用 *）。\n"
            "2. 以 x 为唯一的自变量，形如 sin(x)、x**2 + 2*x + 1。\n"
            "3. 只输出表达式本身，不要 y=、不要引号、不要任何解释文字。\n"
            "4. 二次函数 y=a*x**2 中，|a| 越大开口越窄、|a| 越小开口越宽；"
            "“开口收窄/变窄”应让 |a| 增大（如乘 2），“开口变宽/放大”应让 |a| 减小（如乘 0.5）。\n\n"
            f"当前函数表达式：{current_expr}\n"
            f"修改要求：{instruction}\n"
        )

        for _ in range(2):  # 失败重试一次
            try:
                resp = requests.post(
                    DEEPSEEK_API_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": DEEPSEEK_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                return None

            # 清洗模型输出：去掉代码块围栏、y= 前缀、多余引号
            content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            content = content.strip().strip("\"'`")
            content = re.sub(r"^y\s*=\s*", "", content).strip()
            if not content:
                continue

            # 本地验证：表达式必须能对 x 计算出同长度结果
            normalized = _normalize_expr(content)
            probe = np.linspace(-5, 5, 101)
            try:
                y = _safe_eval(normalized, {**SAFE_FUNCS, "x": probe})
                y = np.asarray(y, dtype=float)
                if y.size == probe.size:
                    return content
            except Exception:
                continue
        return None

    @staticmethod
    def _parse_transform(text):
        """把一句中文描述解析成 (变换类型, 参数字典)，无法理解返回 None"""
        m = re.search(r"[-+]?\d*\.?\d+", text)
        num = float(m.group()) if m else None

        # 旋转（顺时针为负角度）
        if "旋转" in text or "顺时针" in text or "逆时针" in text:
            angle = num if num is not None else 90.0
            if "顺时针" in text:
                angle = -angle
            return "rotate", {"angle": angle}

        # 关于原点对称 = 旋转 180 度
        if "原点" in text:
            return "rotate", {"angle": 180.0}

        # 翻折 / 对称
        if ("对称" in text or "翻折" in text) and "x轴" in text:
            return "flip_x", {}
        if ("对称" in text or "翻折" in text) and "y轴" in text:
            return "flip_y", {}

        # 平移
        if "平移" in text or "移" in text:
            d = num if num is not None else 1.0
            if "右" in text:
                return "right", {"d": d}
            if "左" in text:
                return "left", {"d": d}
            if "上" in text:
                return "up", {"d": d}
            if "下" in text:
                return "down", {"d": d}
            return None

        # 伸缩
        if any(k in text for k in ("拉伸", "压缩", "扩大", "缩小", "放大", "倍")):
            k = num if num is not None else 2.0
            if any(k in text for k in ("压缩", "缩小")):  # 压缩 -> 除以倍数
                k = 1.0 / k
            if "纵" in text or "y" in text or "上下" in text:
                return "scale_y", {"k": k}
            if "横" in text or "x" in text:
                return "scale_x", {"k": k}
            return "scale_y", {"k": k}  # 默认按纵向

        return None

    @staticmethod
    def _apply_transform(x, y, kind, params):
        """对曲线上的点做几何变换，返回 (新x, 新y)"""
        if kind == "right":
            x = x + params["d"]
        elif kind == "left":
            x = x - params["d"]
        elif kind == "up":
            y = y + params["d"]
        elif kind == "down":
            y = y - params["d"]
        elif kind == "scale_x":
            x = x * params["k"]
        elif kind == "scale_y":
            y = y * params["k"]
        elif kind == "flip_x":
            y = -y
        elif kind == "flip_y":
            x = -x
        elif kind == "rotate":
            t = np.radians(params["angle"])
            new_x = x * np.cos(t) - y * np.sin(t)
            new_y = x * np.sin(t) + y * np.cos(t)
            x, y = new_x, new_y
        return x, y

    def clear_all(self):
        """清空所有曲线"""
        self.curves.clear()
        self._pending_expr = None
        self.apply_btn.configure(state="disabled")
        self._redraw()
        self._chat_append("系统", "已清除全部曲线")


if __name__ == "__main__":
    root = tk.Tk()
    app = FunctionPlotter(root)
    root.mainloop()
