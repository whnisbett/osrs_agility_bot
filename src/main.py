import win32gui
from PIL import ImageGrab, Image
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger("Agility Bot Logger")


class Bot:
    def __init__(self):
        # Initialize detector, have it look for green. If no green, look for red. If found red, look for MoG.
        return

    def _get_rl_screenshot(self):
        win_handles = self._get_window_handles()
        rl_handle_id = self._find_rl_handle(win_handles)
        win32gui.SetForegroundWindow(rl_handle_id)
        bbox = win32gui.GetWindowRect(rl_handle_id)
        img = ImageGrab.grab(bbox)
        img = np.array(img)
        return img

    def _find_rl_handle(self, win_handles):
        rl_handle = [
            (handle_id, title)
            for handle_id, title in win_handles
            if title == "RuneLite - BilluMays"
        ]
        if len(rl_handle) == 0:
            log.error("RuneLite does not appear to be opened.")
        rl_handle = rl_handle[0]
        rl_handle_id = rl_handle[0]
        return rl_handle_id

    def _get_window_handles(self):
        win_handles = []

        def window_enum_handler(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                win_handles.append((hwnd, win32gui.GetWindowText(hwnd)))

        win32gui.EnumWindows(window_enum_handler, None)

        return win_handles