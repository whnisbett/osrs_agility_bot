import win32gui
import cv2
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

    def get_rl_screenshot(self):
        win_handles = self.get_window_handles()
        rl_handle_id = self.find_rl_handle(win_handles)
        win32gui.SetForegroundWindow(rl_handle_id)
        bbox = win32gui.GetWindowRect(rl_handle_id)
        img = ImageGrab.grab(bbox)
        img = np.array(img)
        return img

    def find_rl_handle(self, win_handles):
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

    def get_window_handles(self):
        win_handles = []

        def window_enum_handler(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                win_handles.append((hwnd, win32gui.GetWindowText(hwnd)))

        win32gui.EnumWindows(window_enum_handler, None)

        return win_handles

    def remove_hud(self, img):
        img = self.remove_border(img)
        img = self.remove_chatbox(img)
        img = self.remove_minimap(img)
        img = self.remove_toolbar(img)
        img = self.remove_info(img)

        return img

    def remove_border(self, img):
        img[0:30, :, :] = 0
        img[:, 2450:, :] = 0
        return img

    def remove_chatbox(self, img):
        img[1240:, :520, :] = 0
        return img

    def remove_minimap(self, img):
        img[:200, 2240:, :] = 0
        return img

    def remove_toolbar(self, img):
        img[1300:, 2020:, :] = 0
        return img

    def remove_info(self, img):
        img[:175, :150, :] = 0
        return img


class Detector:
    def __init__(self):
        self.img = None
        self.has_mog = None
        self.config = {
            "thresh": 200,
            "maxval": 255,
            "kernel_size": (20, 20),
            "epsilon_frac": 0.05,
            "bbox_area_thresh": 2000,
            "bbox_adjustment": 20
        }
        return

    def __call__(self, img):
        self.img = img
        self.update()
        return

    def update(self):
        # is red or is green (change has_mog based on this)
        #

        return
    def get_clickbox_bboxes(self, img):
        """
        Find bounding boxes for clickable areas.
        """
        thresh_img = self.preprocess_image(img)
        bboxes = self.find_shape_bboxes(thresh_img)
        bboxes = self.remove_extra_bboxes(bboxes)
        bboxes = self.adjust_bboxes(bboxes)
        if len(bboxes) == 0:
            log.warning("No clickboxes detected.")
        return bboxes

    def preprocess_image(self, img):
        """
        Preprocess image for detecting clickboxes and MoGs
        """
        # Apply binary threshold to image to locate highly saturated objects (includes clickboxes and MoGs)
        _, thresh_img = cv2.threshold(
            img,
            self.config["thresh"],
            self.config["maxval"],
            type=cv2.THRESH_BINARY,
        )
        # Expand borders of highly saturated objects to form continuous contours
        img_blur = cv2.blur(thresh_img, self.config["kernel_size"])
        img_blur[img_blur.nonzero()] = self.config["maxval"]
        return img_blur

    def find_shape_bboxes(self, thresh_img):
        """
        Generates contours and creates bounding boxes for each.
        """
        # Locate each contour in the image
        conts, _ = cv2.findContours(
            thresh_img, 
            mode=cv2.RETR_EXTERNAL, # Get only external contours (i.e. outer edges)
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        # Create a bounding box around each of the contours
        bboxes = []
        for cont in conts:
            arc_length = cv2.arcLength(cont, closed=True)
            shape = cv2.approxPolyDP(
                cont, epsilon=self.config["epsilon_frac"] * arc_length, closed=True
            )
            bbox = cv2.boundingRect(shape)
            bboxes.append(bbox)
        return bboxes

    def remove_extra_bboxes(self, bboxes):
        """
        Removes bounding boxes that do no have a minimum area as defined in the config.
        """
        bboxes = [
            bbox
            for bbox in bboxes
            if self.compute_bbox_area(bbox) > self.config["bbox_area_thresh"]
        ]
        return bboxes

    def compute_bbox_area(self, bbox):
        """
        Computes area of bounding box from its (x, y, l, w)-style coordinates.
        """
        return bbox[2] * bbox[3]

    def adjust_bboxes(self, bboxes):
        """
        Shrink bounding boxes around their centers by some number of pixels defined in the config.
        """
        adjustment = self.config["bbox_adjustment"]
        adj_bboxes = []
        for bbox in bboxes:
            x, y, l, w = bbox
            x = round(x + adjustment/2)
            y = round(y + adjustment/2)
            l = round(l - adjustment)
            w = round(w - adjustment)
            bbox = (x, y, l, w)
            adj_bboxes.append(bbox)
        return adj_bboxes
