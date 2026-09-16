"""Windows coordinates and bounded input for the user's screen-player app."""

import ctypes
from ctypes import wintypes
import sys
import time


def enable_dpi_awareness():
    if sys.platform != "win32":
        raise RuntimeError("The screen player currently supports Windows only")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KeyboardInput(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class InputUnion(ctypes.Union):
    _fields_ = [("mi", MouseInput), ("ki", KeyboardInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("data", InputUnion)]


class WindowsInput:
    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("Windows is required")
        self.api = ctypes.WinDLL("user32", use_last_error=True)
        self.api.GetSystemMetrics.argtypes = [ctypes.c_int]
        self.api.GetSystemMetrics.restype = ctypes.c_int
        self.api.WindowFromPoint.argtypes = [wintypes.POINT]
        self.api.WindowFromPoint.restype = wintypes.HWND
        self.api.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        self.api.GetAncestor.restype = wintypes.HWND
        self.api.GetForegroundWindow.restype = wintypes.HWND
        self.api.SetForegroundWindow.argtypes = [wintypes.HWND]
        self.api.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
        self.api.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.api.GetAsyncKeyState.restype = wintypes.SHORT
        self.api.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
        self.api.SendInput.restype = wintypes.UINT

    def desktop(self):
        return tuple(self.api.GetSystemMetrics(index) for index in (76, 77, 78, 79))

    def window_at(self, point):
        return self.api.GetAncestor(self.api.WindowFromPoint(wintypes.POINT(*point)), 2)

    def foreground(self):
        return self.api.GetForegroundWindow()

    def stopped(self, stop):
        return stop.is_set() or bool(self.api.GetAsyncKeyState(0x77) & 0x8000)  # F8

    def move_pointer(self, point, target, stop):
        if self.stopped(stop):
            raise InterruptedError("Stopped with F8")
        if self.window_at(point) != target or self.foreground() != target:
            raise InterruptedError("The selected board window is no longer in front")
        left, top, width, height = self.desktop()
        x, y = point
        if not left <= x < left + width or not top <= y < top + height:
            raise ValueError("Pointer would be outside the desktop")
        movement = Input(0, InputUnion(mi=MouseInput(round((x-left)*65535/(width-1)),
                                                   round((y-top)*65535/(height-1)), 0,
                                                   0x8000 | 0x4000 | 0x0001, 0, 0)))
        if self.api.SendInput(1, ctypes.byref(movement), ctypes.sizeof(Input)) != 1:
            raise OSError("Windows could not move the pointer")

    def park_pointer(self, area, target, stop):
        # Keep cursor highlights and hover effects out of the board capture.
        x, y = (area.left + area.right) // 2, (area.top + area.bottom) // 2
        left, top, width, height = self.desktop()
        for point in ((x, area.top - 64), (x, area.bottom + 64),
                      (area.left - 64, y), (area.right + 64, y)):
            if (left <= point[0] < left + width and top <= point[1] < top + height
                    and self.window_at(point) == target):
                self.move_pointer(point, target, stop)
                return

    def click(self, point, target, stop):
        self.move_pointer(point, target, stop)
        time.sleep(0.05)
        if self.stopped(stop) or self.window_at(point) != target or self.foreground() != target:
            raise InterruptedError("Input stopped before clicking")
        # Send down/up together so stopping cannot leave the mouse button held.
        events = (Input * 2)(Input(0, InputUnion(mi=MouseInput(0, 0, 0, 2, 0, 0))),
                             Input(0, InputUnion(mi=MouseInput(0, 0, 0, 4, 0, 0))))
        if self.api.SendInput(2, events, ctypes.sizeof(Input)) != 2:
            release = Input(0, InputUnion(mi=MouseInput(0, 0, 0, 4, 0, 0)))
            self.api.SendInput(1, ctypes.byref(release), ctypes.sizeof(Input))
            raise OSError("Windows could not click. The target may require elevated access.")
