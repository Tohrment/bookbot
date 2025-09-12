import threading
import time
import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# --------------------------
# Action Model
# --------------------------

SUPPORTED_ACTIONS = [
    "WAIT",
    "SCREENSHOT",
    "LOAD_TEMPLATE",
    "FIND_TEMPLATE",
    "CLICK_MATCH",
    "CLICK_AT",
    "MOVE_MOUSE",
    "CHECK_PIXEL",
    "SET_VAR",
    "IF_GOTO",
    "PRESS_KEY",
    "TYPE_TEXT",
]

@dataclass
class BotAction:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {"type": self.type, "params": self.params}

    @staticmethod
    def from_dict(d: Dict[str, Any]):
        return BotAction(type=d["type"], params=d.get("params", {}))


# --------------------------
# Bot Runtime
# --------------------------

class BotRuntime:
    def __init__(self):
        self.vars: Dict[str, Any] = {}
        self.templates: Dict[str, np.ndarray] = {}
        self.last_screenshot_bgr: Optional[np.ndarray] = None
        self.last_match: Optional[Dict[str, Any]] = None
        self.stop_requested = False
        self.lock = threading.Lock()

    def _debug(self, msg: str):
        # print(msg)  # enable if you want console logs
        pass

    def request_stop(self):
        with self.lock:
            self.stop_requested = True

    def clear_stop(self):
        with self.lock:
            self.stop_requested = False

    def should_stop(self) -> bool:
        with self.lock:
            return self.stop_requested

    def take_screenshot(self) -> np.ndarray:
        self._debug("Taking screenshot")
        img = pyautogui.screenshot()
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self.last_screenshot_bgr = bgr
        return bgr

    def load_template(self, name: str, path: str) -> bool:
        self._debug(f"Loading template {name} from {path}")
        if not os.path.isfile(path):
            return False
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            return False
        self.templates[name] = im
        return True

    def find_template(self, name: str, threshold: float = 0.85, method: int = cv2.TM_CCOEFF_NORMED) -> Optional[Dict[str, Any]]:
        self._debug(f"Finding template {name} with threshold {threshold}")
        if self.last_screenshot_bgr is None:
            self.take_screenshot()
        if name not in self.templates:
            return None
        src = self.last_screenshot_bgr
        tmpl = self.templates[name]
        res = cv2.matchTemplate(src, tmpl, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            score = 1.0 - float(min_val)
            loc = min_loc
        else:
            score = float(max_val)
            loc = max_loc
        h, w = tmpl.shape[:2]
        if score >= threshold:
            self.last_match = {"name": name, "x": int(loc[0]), "y": int(loc[1]), "w": int(w), "h": int(h), "score": score}
            return self.last_match
        self.last_match = None
        return None

    def click_match(self, button: str = "left", clicks: int = 1, interval: float = 0.05, center: bool = True, offset: Tuple[int, int] = (0, 0)) -> bool:
        self._debug("Click match")
        if not self.last_match:
            return False
        x = self.last_match["x"] + (self.last_match["w"] // 2 if center else 0) + int(offset[0])
        y = self.last_match["y"] + (self.last_match["h"] // 2 if center else 0) + int(offset[1])
        pyautogui.click(x, y, clicks=clicks, button=button, interval=interval)
        return True

    def click_at(self, x: int, y: int, button: str = "left", clicks: int = 1, interval: float = 0.05) -> None:
        self._debug(f"Click at {x},{y}")
        pyautogui.click(int(x), int(y), clicks=clicks, interval=interval, button=button)

    def move_mouse(self, x: int, y: int, duration: float = 0.2) -> None:
        self._debug(f"Move mouse to {x},{y}")
        pyautogui.moveTo(int(x), int(y), duration=max(0.0, float(duration)))

    def press_key(self, key: str) -> None:
        self._debug(f"Press key {key}")
        pyautogui.press(key)

    def type_text(self, text: str, interval: float = 0.02) -> None:
        self._debug(f"Type text '{text}'")
        pyautogui.typewrite(text, interval=max(0.0, float(interval)))

    def check_pixel(self, x: int, y: int, r: int, g: int, b: int, tolerance: int = 10) -> bool:
        self._debug(f"Check pixel at {x},{y} ~ ({r},{g},{b}) tol {tolerance}")
        if self.last_screenshot_bgr is None:
            self.take_screenshot()
        h, w = self.last_screenshot_bgr.shape[:2]
        xi, yi = int(x), int(y)
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            return False
        bgr = self.last_screenshot_bgr[yi, xi].tolist()
        br, bg, bb = int(bgr[0]), int(bgr[1]), int(bgr[2])
        # convert target RGB to BGR for comparison
        tr, tg, tb = int(b), int(g), int(r)
        return abs(br - tr) <= tolerance and abs(bg - tg) <= tolerance and abs(bb - tb) <= tolerance

    def set_var(self, name: str, value: Any):
        self._debug(f"Set var {name}={value}")
        self.vars[name] = value

    def get_var(self, name: str, default=None):
        return self.vars.get(name, default)


# --------------------------
# GUI
# --------------------------

class BotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Recognition Automation Bot")
        self.geometry("1200x750")
        self.runtime = BotRuntime()
        self.actions: List[BotAction] = []
        self.runner_thread: Optional[threading.Thread] = None

        # UI Layout
        self._build_menu()
        self._build_left_panel()
        self._build_canvas_panel()
        self._build_right_panel()
        self._bind_canvas_events()

        self.update_preview()  # initialize canvas

    # ---- UI Builders ----

    def _build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.new_script)
        filemenu.add_command(label="Open...", command=self.load_script)
        filemenu.add_command(label="Save As...", command=self.save_script)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

    def _build_left_panel(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Actions").pack(anchor="w")
        self.actions_list = tk.Listbox(left, width=40, height=28)
        self.actions_list.pack(fill=tk.Y, expand=False)

        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btns, text="Up", command=self.move_action_up).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Down", command=self.move_action_down).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Delete", command=self.delete_action).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Test Selected", command=self.test_selected_action).pack(side=tk.LEFT, padx=2)

        runrow = ttk.Frame(left)
        runrow.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(runrow, text="Run", command=self.run_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(runrow, text="Stop", command=self.stop_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(runrow, text="Screenshot", command=self.cmd_screenshot).pack(side=tk.LEFT, padx=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(left, text="Templates").pack(anchor="w")
        self.templates_list = tk.Listbox(left, width=40, height=10)
        self.templates_list.pack(fill=tk.Y, expand=False)
        tbtns = ttk.Frame(left)
        tbtns.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(tbtns, text="Load Template...", command=self.load_template_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(tbtns, text="Remove Template", command=self.remove_selected_template).pack(side=tk.LEFT, padx=2)

    def _build_canvas_panel(self):
        center = ttk.Frame(self, padding=8)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(center, width=800, height=600, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        status = ttk.Frame(center)
        status.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)

    def _build_right_panel(self):
        right = ttk.Frame(self, padding=8)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(right, text="Add Action").pack(anchor="w")
        self.action_type_var = tk.StringVar(value=SUPPORTED_ACTIONS[0])
        ttk.Combobox(right, values=SUPPORTED_ACTIONS, textvariable=self.action_type_var, state="readonly", width=28).pack(anchor="w", pady=(0, 6))

        self.param_frame = ttk.Frame(right)
        self.param_frame.pack(fill=tk.X)
        self.param_entries: Dict[str, tk.Entry] = {}
        self._refresh_param_inputs()

        ttk.Button(right, text="Add", command=self.add_action).pack(anchor="w", pady=(6, 12))

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        ttk.Label(right, text="Pixel Picker").pack(anchor="w", pady=(4, 0))
        self.pick_info = tk.StringVar(value="Click on the canvas to get coordinates and color.")
        ttk.Label(right, textvariable=self.pick_info, wraplength=260).pack(anchor="w")

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        ttk.Label(right, text="Variables").pack(anchor="w")
        self.vars_text = tk.Text(right, width=36, height=18)
        self.vars_text.pack()

        self.action_type_var.trace_add("write", lambda *_: self._refresh_param_inputs())

    def _bind_canvas_events(self):
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_move)

    # ---- Helpers ----

    def set_status(self, text: str):
        self.status_var.set(text)
        self.update_idletasks()

    def update_preview(self, rectangles: List[Tuple[int, int, int, int]] = None):
        rectangles = rectangles or []
        bgr = self.runtime.last_screenshot_bgr
        if bgr is None:
            img = np.zeros((600, 800, 3), dtype=np.uint8)
        else:
            img = bgr.copy()
        # draw rectangles
        for (x, y, w, h) in rectangles:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        # Fit to canvas while preserving aspect ratio
        cwidth = int(self.canvas.winfo_width() or 800)
        cheight = int(self.canvas.winfo_height() or 600)
        pil.thumbnail((cwidth, cheight), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cwidth // 2, cheight // 2, image=self.tk_img)

        # update vars pane
        pretty = json.dumps({"vars": self.runtime.vars, "last_match": self.runtime.last_match}, indent=2)
        self.vars_text.delete("1.0", tk.END)
        self.vars_text.insert(tk.END, pretty)

    def on_canvas_click(self, event):
        if self.runtime.last_screenshot_bgr is None:
            return
        # Map click to image coordinate by assuming preview is scaled with thumbnail
        # Approximate by scaling factor between original and displayed image.
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_h, img_w = self.runtime.last_screenshot_bgr.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        x0 = (canvas_w - disp_w) // 2
        y0 = (canvas_h - disp_h) // 2
        x_img = int((event.x - x0) / scale)
        y_img = int((event.y - y0) / scale)
        if x_img < 0 or y_img < 0 or x_img >= img_w or y_img >= img_h:
            return
        b, g, r = self.runtime.last_screenshot_bgr[y_img, x_img].tolist()
        self.pick_info.set(f"Canvas: ({event.x},{event.y})  Image: ({x_img},{y_img})  Color RGB: ({r},{g},{b})")
        self.set_status(f"Picked ({x_img},{y_img}) RGB({r},{g},{b})")

    def on_canvas_move(self, event):
        # optional live coordinate display (kept simple)
        pass

    def _refresh_param_inputs(self):
        for child in self.param_frame.winfo_children():
            child.destroy()
        self.param_entries.clear()

        t = self.action_type_var.get()

        def add_field(label, default=""):
            ttk.Label(self.param_frame, text=label).pack(anchor="w")
            e = ttk.Entry(self.param_frame, width=30)
            e.insert(0, str(default))
            e.pack(anchor="w", pady=(0, 4))
            self.param_entries[label] = e

        if t == "WAIT":
            add_field("seconds", "0.5")
        elif t == "SCREENSHOT":
            pass
        elif t == "LOAD_TEMPLATE":
            add_field("name", "target1")
            add_field("path", "")
        elif t == "FIND_TEMPLATE":
            add_field("name", "target1")
            add_field("threshold", "0.85")
        elif t == "CLICK_MATCH":
            add_field("button", "left")
            add_field("clicks", "1")
            add_field("interval", "0.05")
            add_field("center", "true")
            add_field("offset_x", "0")
            add_field("offset_y", "0")
        elif t == "CLICK_AT":
            add_field("x", "100")
            add_field("y", "100")
            add_field("button", "left")
            add_field("clicks", "1")
            add_field("interval", "0.05")
        elif t == "MOVE_MOUSE":
            add_field("x", "200")
            add_field("y", "200")
            add_field("duration", "0.2")
        elif t == "CHECK_PIXEL":
            add_field("x", "100")
            add_field("y", "100")
            add_field("r", "255")
            add_field("g", "255")
            add_field("b", "255")
            add_field("tolerance", "10")
            add_field("set_var", "pixel_ok")
        elif t == "SET_VAR":
            add_field("name", "flag1")
            add_field("value", "true")
        elif t == "IF_GOTO":
            add_field("expr", "vars.get('pixel_ok') == True and (last_match is not None)")
            add_field("index", "0")
        elif t == "PRESS_KEY":
            add_field("key", "enter")
        elif t == "TYPE_TEXT":
            add_field("text", "hello world")
            add_field("interval", "0.02")

    def _collect_params(self) -> Dict[str, Any]:
        params = {}
        for k, e in self.param_entries.items():
            v = e.get()
            # try to auto-cast
            if v.lower() in ("true", "false"):
                params[k] = (v.lower() == "true")
            else:
                try:
                    if "." in v:
                        params[k] = float(v)
                    else:
                        params[k] = int(v)
                except ValueError:
                    params[k] = v
        return params

    # ---- Actions management ----

    def add_action(self):
        t = self.action_type_var.get()
        params = self._collect_params()
        act = BotAction(type=t, params=params)
        self.actions.append(act)
        self.actions_list.insert(tk.END, self._action_to_text(act))

    def delete_action(self):
        idx = self._selected_action_index()
        if idx is None:
            return
        del self.actions[idx]
        self.actions_list.delete(idx)

    def move_action_up(self):
        idx = self._selected_action_index()
        if idx is None or idx == 0:
            return
        self.actions[idx - 1], self.actions[idx] = self.actions[idx], self.actions[idx - 1]
        self._refresh_actions_list()
        self.actions_list.selection_set(idx - 1)

    def move_action_down(self):
        idx = self._selected_action_index()
        if idx is None or idx >= len(self.actions) - 1:
            return
        self.actions[idx + 1], self.actions[idx] = self.actions[idx], self.actions[idx + 1]
        self._refresh_actions_list()
        self.actions_list.selection_set(idx + 1)

    def _refresh_actions_list(self):
        self.actions_list.delete(0, tk.END)
        for act in self.actions:
            self.actions_list.insert(tk.END, self._action_to_text(act))

    def _selected_action_index(self) -> Optional[int]:
        sel = self.actions_list.curselection()
        if not sel:
            return None
        return int(sel[0])

    def _action_to_text(self, act: BotAction) -> str:
        return f"{act.type}  {json.dumps(act.params)}"

    # ---- Script persistence ----

    def new_script(self):
        if not self._confirm_discard():
            return
        self.actions.clear()
        self._refresh_actions_list()
        self.runtime.vars.clear()
        self.runtime.last_match = None
        self.set_status("New script")

    def save_script(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        data = [a.to_dict() for a in self.actions]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.set_status(f"Saved: {path}")

    def load_script(self):
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.actions = [BotAction.from_dict(d) for d in data]
            self._refresh_actions_list()
            self.set_status(f"Loaded: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load script:\n{e}")

    def _confirm_discard(self) -> bool:
        if not self.actions:
            return True
        return messagebox.askyesno("Confirm", "Discard current script?")

    # ---- Templates ----

    def load_template_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        name = os.path.splitext(os.path.basename(path))[0]
        if self.runtime.load_template(name, path):
            self.templates_list.insert(tk.END, f"{name} -> {path}")
            self.set_status(f"Loaded template '{name}'")
        else:
            messagebox.showerror("Error", "Failed to load template.")

    def remove_selected_template(self):
        idxs = self.templates_list.curselection()
        if not idxs:
            return
        idx = idxs[0]
        item = self.templates_list.get(idx)
        name = item.split(" -> ")[0]
        if name in self.runtime.templates:
            del self.runtime.templates[name]
        self.templates_list.delete(idx)

    # ---- Commands ----

    def cmd_screenshot(self):
        self.runtime.take_screenshot()
        rects = []
        if self.runtime.last_match:
            m = self.runtime.last_match
            rects.append((m["x"], m["y"], m["w"], m["h"]))
        self.update_preview(rectangles=rects)

    def test_selected_action(self):
        idx = self._selected_action_index()
        if idx is None:
            return
        self._execute_action(idx, test_mode=True)

    def run_script(self):
        if self.runner_thread and self.runner_thread.is_alive():
            return
        self.runtime.clear_stop()
        self.runner_thread = threading.Thread(target=self._runner_loop, daemon=True)
        self.runner_thread.start()

    def stop_script(self):
        self.runtime.request_stop()
        self.set_status("Stopping...")

    # ---- Runner ----

    def _runner_loop(self):
        i = 0
        self.set_status("Running")
        while i < len(self.actions):
            if self.runtime.should_stop():
                self.set_status("Stopped")
                return
            jumped = self._execute_action(i)
            # _execute_action may return a jump index (int) or None
            if isinstance(jumped, int):
                i = max(0, min(len(self.actions) - 1, jumped))
            else:
                i += 1
        self.set_status("Done")

    def _execute_action(self, index: int, test_mode: bool = False) -> Optional[int]:
        if index < 0 or index >= len(self.actions):
            return None
        act = self.actions[index]
        r = self.runtime
        p = act.params

        try:
            if act.type == "WAIT":
                secs = float(p.get("seconds", 0.5))
                self.set_status(f"[{index}] WAIT {secs}s")
                time.sleep(max(0.0, secs))

            elif act.type == "SCREENSHOT":
                self.set_status(f"[{index}] SCREENSHOT")
                r.take_screenshot()
                self.update_preview()

            elif act.type == "LOAD_TEMPLATE":
                name = str(p.get("name", "template"))
                path = str(p.get("path", ""))
                self.set_status(f"[{index}] LOAD_TEMPLATE {name}")
                ok = r.load_template(name, path)
                if ok and test_mode:
                    self.templates_list.insert(tk.END, f"{name} -> {path}")
                if not ok:
                    messagebox.showwarning("LOAD_TEMPLATE", f"Failed to load template '{name}' at '{path}'")

            elif act.type == "FIND_TEMPLATE":
                name = str(p.get("name", "template"))
                threshold = float(p.get("threshold", 0.85))
                self.set_status(f"[{index}] FIND_TEMPLATE {name} thr={threshold}")
                m = r.find_template(name, threshold=threshold)
                rects = []
                if m:
                    rects.append((m["x"], m["y"], m["w"], m["h"]))
                    r.set_var("last_match_score", m["score"])
                    r.set_var("last_match_name", m["name"])
                    r.set_var("last_match_xy", (m["x"], m["y"]))
                else:
                    r.set_var("last_match_score", None)
                    r.set_var("last_match_name", None)
                    r.set_var("last_match_xy", None)
                self.update_preview(rectangles=rects)

            elif act.type == "CLICK_MATCH":
                button = str(p.get("button", "left"))
                clicks = int(p.get("clicks", 1))
                interval = float(p.get("interval", 0.05))
                center = bool(p.get("center", True))
                ox = int(p.get("offset_x", 0))
                oy = int(p.get("offset_y", 0))
                self.set_status(f"[{index}] CLICK_MATCH {button} x{clicks}")
                ok = r.click_match(button=button, clicks=clicks, interval=interval, center=center, offset=(ox, oy))
                if not ok:
                    messagebox.showinfo("CLICK_MATCH", "No last match to click.")

            elif act.type == "CLICK_AT":
                x = int(p.get("x", 0))
                y = int(p.get("y", 0))
                button = str(p.get("button", "left"))
                clicks = int(p.get("clicks", 1))
                interval = float(p.get("interval", 0.05))
                self.set_status(f"[{index}] CLICK_AT {x},{y}")
                r.click_at(x, y, button=button, clicks=clicks, interval=interval)

            elif act.type == "MOVE_MOUSE":
                x = int(p.get("x", 0))
                y = int(p.get("y", 0))
                duration = float(p.get("duration", 0.2))
                self.set_status(f"[{index}] MOVE_MOUSE {x},{y}")
                r.move_mouse(x, y, duration=duration)

            elif act.type == "CHECK_PIXEL":
                x = int(p.get("x", 0))
                y = int(p.get("y", 0))
                r0 = int(p.get("r", 255))
                g0 = int(p.get("g", 255))
                b0 = int(p.get("b", 255))
                tol = int(p.get("tolerance", 10))
                varname = str(p.get("set_var", "pixel_ok"))
                self.set_status(f"[{index}] CHECK_PIXEL {x},{y}")
                ok = r.check_pixel(x, y, r0, g0, b0, tol)
                r.set_var(varname, ok)
                self.update_preview()

            elif act.type == "SET_VAR":
                name = str(p.get("name", "flag"))
                value = p.get("value", "")
                # cast basic literals
                if isinstance(value, str):
                    lv = value.lower()
                    if lv in ("true", "false"):
                        value = (lv == "true")
                    else:
                        try:
                            if "." in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass
                self.set_status(f"[{index}] SET_VAR {name}={value}")
                r.set_var(name, value)
                self.update_preview()

            elif act.type == "IF_GOTO":
                expr = str(p.get("expr", "False"))
                target = int(p.get("index", 0))
                env = {"vars": r.vars, "last_match": r.last_match}
                self.set_status(f"[{index}] IF_GOTO -> {target}")
                try:
                    result = bool(eval(expr, {"__builtins__": {}}, env))
                except Exception:
                    result = False
                if result:
                    return target  # jump
                # else continue

            elif act.type == "PRESS_KEY":
                key = str(p.get("key", "enter"))
                self.set_status(f"[{index}] PRESS_KEY {key}")
                r.press_key(key)

            elif act.type == "TYPE_TEXT":
                txt = str(p.get("text", ""))
                interval = float(p.get("interval", 0.02))
                self.set_status(f"[{index}] TYPE_TEXT '{txt}'")
                r.type_text(txt, interval=interval)

            else:
                messagebox.showwarning("Unknown Action", f"Unknown action type: {act.type}")

        except Exception as e:
            messagebox.showerror("Action Error", f"Error at action #{index} ({act.type}):\n{e}")

        return None

# ---- Main ----

if __name__ == "__main__":
    # PyAutoGUI failsafe corner override if needed:
    # pyautogui.FAILSAFE = True  # move mouse to top-left to abort
    app = BotGUI()
    app.mainloop()
