from __future__ import annotations

import platform
import tkinter
from tkinter import filedialog


def _ensure_tkinter_root() -> tkinter.Tk:
    root = tkinter.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    return root


def pick_windows_folder() -> str:
    system = platform.system()
    if system not in {"Windows", "Darwin", "Linux"}:
        raise ValueError(f"不支持的操作系统: {system}")
    return _pick_folder_tkinter()


def _pick_folder_tkinter() -> str:
    root = _ensure_tkinter_root()
    try:
        path = filedialog.askdirectory(title="选择要分析的仓库目录")
        if not path:
            raise ValueError("已取消文件夹选择")
        return path
    finally:
        root.destroy()
