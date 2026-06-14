import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

# Shared Resources with Mutex Lock for Concurrency
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input': 0.0,
    'acceleration_input': 1.0,
    'sent_steering_input': 0.0,
    'sent_acceleration_input': 1.0,
    'drive_mode': 'search',
    'debug_frame': None,
    'back_debug_frame': None
}
data_lock = threading.Lock()
is_running = True
last_status_print = 0.0

# Vision and control tuning
ROI_START = 0.30
PATH_THRESHOLD = 160
EDGE_LOW = 60
EDGE_HIGH = 150
TEST_THROTTLE = 1.0
TOKEN_MIN_AREA = 100
FAR_TOKEN_MIN_AREA = 30
HAZARD_MIN_Y = 0.14
CENTER_LINE_TOLERANCE = 0.04
CENTER_LINE_SOFTNESS = 0.06
CHASE_BACK_CENTER_TOLERANCE = 0.10
CHASE_BACK_HOLD_TIME = 0.50
CHASE_BACK_MIN_AREA = 28
CHASE_BACK_FAR_MIN_AREA = 8
CHASE_BACK_NEAR_MIN_AREA = 55
CHASE_BACK_NEAR_MIN_Y = 0.46
CHASE_BACK_ROI_TOP = 0.42
CHASE_BACK_ROI_BOTTOM = 0.94
CHASE_BACK_BLUE_MIN = 70
CHASE_BACK_BLUE_MAX = 230
CHASE_BACK_GREEN_MIN = 85
CHASE_BACK_GREEN_MAX = 235
CHASE_BACK_RED_MIN = 0
CHASE_BACK_RED_MAX = 120
CHASE_BACK_HUE_MIN = 75
CHASE_BACK_HUE_MAX = 102
CHASE_BACK_SAT_MIN = 60
CHASE_BACK_VAL_MIN = 45
CHASE_BACK_MAX_BG_DIFF = 75
POLICE_HOLD_TIME = 0.40
POLICE_EVENT_DURATION = 10.0
POLICE_RED_CLEAR_CONFIRM_TIME = 0.20
POLICE_ROI_TOP = ROI_START
POLICE_ROI_BOTTOM = 0.82
POLICE_ROI_LEFT = 0.18
POLICE_ROI_RIGHT = 0.82
POLICE_CENTER_TOLERANCE = 0.34
POLICE_FAR_MIN_AREA = 14
POLICE_MIN_AREA = 42
POLICE_NEAR_MIN_AREA = 70
POLICE_NEAR_MIN_Y = 0.40
POLICE_MIN_BOX_WIDTH = 14
POLICE_MIN_BOX_HEIGHT = 10
POLICE_FINAL_MIN_WIDTH = 80
POLICE_FINAL_MIN_HEIGHT = 12
POLICE_MAX_BOX_ASPECT = 3.20
POLICE_MIN_NORM_Y = 0.10
POLICE_RED_BGR_MIN = np.array([0, 0, 85], dtype=np.uint8)
POLICE_RED_BGR_MAX = np.array([110, 125, 185], dtype=np.uint8)
POLICE_BLUE_BGR_MIN = np.array([55, 0, 18], dtype=np.uint8)
POLICE_BLUE_BGR_MAX = np.array([150, 105, 115], dtype=np.uint8)
POLICE_RED_HSV_1_MIN = np.array([0, 70, 45], dtype=np.uint8)
POLICE_RED_HSV_1_MAX = np.array([12, 255, 255], dtype=np.uint8)
POLICE_RED_HSV_2_MIN = np.array([168, 70, 45], dtype=np.uint8)
POLICE_RED_HSV_2_MAX = np.array([180, 255, 255], dtype=np.uint8)
POLICE_BLUE_HSV_MIN = np.array([112, 55, 25], dtype=np.uint8)
POLICE_BLUE_HSV_MAX = np.array([145, 255, 170], dtype=np.uint8)
POLICE_MIN_COLOR_PIXELS = 12
POLICE_MIN_COLOR_SHARE = 0.07
POLICE_MIN_HALF_COLOR_PIXELS = 8
POLICE_MIN_HALF_COLOR_SHARE = 0.10
POLICE_MIN_PAIR_FILL_RATIO = 0.16
STEERING_TAP_COOLDOWN = 0.03
GREEN_STEERING_TAP_COOLDOWN = 0.02
STEERING_CONFIRM_CYCLES = 1
LOW_LIGHT_BASELINE_EMA = 0.015
LOW_LIGHT_RATIO_EMA = 0.020
LOW_LIGHT_ENTRY_BRIGHTNESS_RATIO = 0.58
LOW_LIGHT_ENTRY_BRIGHT_PIXEL_RATIO = 0.42
LOW_LIGHT_EXIT_BRIGHTNESS_RATIO = 0.78
LOW_LIGHT_EXIT_BRIGHT_PIXEL_RATIO = 0.68
LOW_LIGHT_ABSOLUTE_BRIGHTNESS = 58.0
LOW_LIGHT_BRIGHT_PIXEL_THRESHOLD = 90
LOW_LIGHT_MAX_HALF_BRIGHTNESS_GAP = 24.0
LOW_LIGHT_MAX_HALF_RATIO_GAP = 0.22
EDGE_MARGIN_RATIO = 0.25
MIN_TOKEN_ASPECT = 0.55
MAX_TOKEN_ASPECT = 1.80
MIN_ROAD_OVERLAP = 0.55
RED_MAX_FILL_RATIO = 0.95
RED_MIN_COMPACTNESS = 0.22
LANE_BOUNDARY_MARGIN = 0.06
RED_STRIPE_MIN_EXTENT = 0.78
RED_STRIPE_MIN_AREA = 140
RED_MAX_MEAN_SATURATION = 210.0
WHITE_MAX_SATURATION = 35
WHITE_MIN_VALUE = 240
YELLOW_BONE_MAX_SATURATION = 45
YELLOW_BONE_MIN_VALUE = 170
YELLOW_RED_EXCLUDE_MIN_H = 26
YELLOW_RED_EXCLUDE_MAX_H = 58
YELLOW_RED_EXCLUDE_MIN_S = 45
YELLOW_RED_EXCLUDE_MIN_V = 105
YELLOW_CORE_MIN_RED = 185
YELLOW_CORE_MIN_GREEN = 135
YELLOW_CORE_MAX_BLUE = 120
YELLOW_CORE_MIN_RG_DIFF = 20
YELLOW_CORE_MAX_RG_DIFF = 90
YELLOW_CORE_MIN_GB_DIFF = 45
RED_MEAN_HUE_MAX = 18
RED_MEAN_HUE_MIN_WRAP = 162
GRAY_MAX_SATURATION = 70
GRAY_MIN_VALUE = 135
GRAY_MAX_VALUE = 220
GRAY_MIN_COMPACTNESS = 0.18
GRAY_MAX_FILL_RATIO = 0.98
GREEN_MIN_Y = 0.02
GREEN_MAX_OFFSET = 0.80
HAZARD_OVERRIDE_THREAT = 0.16
ROAD_LEFT_LIMIT = -0.88
ROAD_RIGHT_LIMIT = 0.88
HAZARD_CLEARANCE = 0.10
HAZARD_GAP_GREEN_BONUS = 0.35
HAZARD_FRONT_PRIORITY_WINDOW = 0.16
HAZARD_SIZE_PRIORITY_WEIGHT = 0.90
GREEN_FRONT_PRIORITY_WINDOW = 0.12
GREEN_SIZE_PRIORITY_WEIGHT = 0.75
GREEN_FRONT_PRIORITY_ADVANTAGE = 0.04
PATH_RELEASE_CENTER_THRESHOLD = 0.15
GREEN_HOLD_CENTER_BAND = 0.06
PATH_LOST_TIMEOUT = 0.75
ROAD_BOUNDARY_ROW_BAND = 6
ROAD_BOUNDARY_MARGIN_PIXELS = 6


# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True

    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH:
                ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.MEDIUM:
                ctypes.windll.kernel32.SetThreadPriority(handle, 0)
            elif self.priority == TaskPriority.LOW:
                ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except Exception:
            pass

        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time

            if sleep_time > 0:
                time.sleep(sleep_time)


# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None


def setup_cameras():
    global front_camera_sock, back_camera_sock

    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False

    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass

        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass

        if not (front_connected and back_connected):
            time.sleep(1)


def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")

    while is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue


# ---------------------------------------------------------
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------
def read_single_camera(sock, window_name, data_key):
    # This function reads the latest frame from the camera socket and stores it in the shared data
    if sock is None:
        return

    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return

        image_length = int.from_bytes(length_bytes, 'little')
        received_bytes = b''
        while len(received_bytes) < image_length and is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet

        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes

        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break

            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return

            image_length = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    break
                received_bytes += packet

            if len(received_bytes) == image_length:
                latest_frame_data = received_bytes

        if latest_frame_data is not None:
            np_arr = np.frombuffer(latest_frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with data_lock:
                    shared_data[data_key] = frame

                frame_to_show = frame
                if data_key == 'latest_front_frame':
                    with data_lock:
                        debug_frame = shared_data['debug_frame']
                    if debug_frame is not None and debug_frame.shape == frame.shape:
                        frame_to_show = debug_frame
                elif data_key == 'latest_back_frame':
                    with data_lock:
                        back_debug_frame = shared_data['back_debug_frame']
                    if back_debug_frame is not None and back_debug_frame.shape == frame.shape:
                        frame_to_show = back_debug_frame

                frame_resized = cv2.resize(frame_to_show, (640, 480))
                cv2.imshow(window_name, frame_resized)
                cv2.waitKey(1)

    except Exception:
        pass


def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')


def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def quit_requested():
    try:
        return keyboard.is_pressed('q')
    except Exception:
        return False


def normalize_x(x_position, width):
    return (x_position / width) * 2.0 - 1.0


def min_token_area_for_y(norm_y):
    # High-up tokens are farther away, so allow smaller blobs there.
    near_weight = clamp((norm_y - 0.18) / 0.72, 0.0, 1.0)
    return FAR_TOKEN_MIN_AREA + ((TOKEN_MIN_AREA - FAR_TOKEN_MIN_AREA) * near_weight)


def build_road_mask(path_mask):
    road_mask = np.zeros_like(path_mask)
    contours, _ = cv2.findContours(path_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(road_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return road_mask


def center_line_overlap(token_x, center_x_norm):
    distance = abs(token_x - center_x_norm)
    if distance <= CENTER_LINE_TOLERANCE:
        return 1.0

    fade_width = max(CENTER_LINE_SOFTNESS, 1e-3)
    return clamp(1.0 - ((distance - CENTER_LINE_TOLERANCE) / fade_width), 0.0, 1.0)


def detect_low_light(frame):
    if not hasattr(detect_low_light, "baseline_brightness"):
        detect_low_light.baseline_brightness = None
        detect_low_light.baseline_bright_ratio = None
        detect_low_light.is_active = False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_top = int(frame.shape[0] * 0.08)
    roi_bottom = int(frame.shape[0] * 0.80)
    gray_roi = gray[roi_top:roi_bottom, :]
    half_width = gray_roi.shape[1] // 2
    left_half = gray_roi[:, :half_width]
    right_half = gray_roi[:, half_width:]

    brightness_mean = float(np.mean(gray_roi))
    bright_pixel_ratio = float(np.mean(gray_roi >= LOW_LIGHT_BRIGHT_PIXEL_THRESHOLD))
    left_brightness = float(np.mean(left_half))
    right_brightness = float(np.mean(right_half))
    left_bright_ratio = float(np.mean(left_half >= LOW_LIGHT_BRIGHT_PIXEL_THRESHOLD))
    right_bright_ratio = float(np.mean(right_half >= LOW_LIGHT_BRIGHT_PIXEL_THRESHOLD))
    half_brightness_gap = abs(left_brightness - right_brightness)
    half_ratio_gap = abs(left_bright_ratio - right_bright_ratio)
    whole_scene_dark = (
        half_brightness_gap <= LOW_LIGHT_MAX_HALF_BRIGHTNESS_GAP and
        half_ratio_gap <= LOW_LIGHT_MAX_HALF_RATIO_GAP
    )

    if detect_low_light.baseline_brightness is None:
        detect_low_light.baseline_brightness = brightness_mean
        detect_low_light.baseline_bright_ratio = bright_pixel_ratio
        return False, brightness_mean, bright_pixel_ratio

    baseline_brightness = detect_low_light.baseline_brightness
    baseline_bright_ratio = detect_low_light.baseline_bright_ratio

    brightness_ratio = brightness_mean / max(baseline_brightness, 1.0)
    bright_ratio_ratio = bright_pixel_ratio / max(baseline_bright_ratio, 1e-4)

    low_light_now = (
        whole_scene_dark and (
            brightness_mean <= LOW_LIGHT_ABSOLUTE_BRIGHTNESS or
            (
                brightness_ratio <= LOW_LIGHT_ENTRY_BRIGHTNESS_RATIO and
                bright_ratio_ratio <= LOW_LIGHT_ENTRY_BRIGHT_PIXEL_RATIO
            )
        )
    )

    if detect_low_light.is_active:
        recovered = (
            not whole_scene_dark or
            brightness_ratio >= LOW_LIGHT_EXIT_BRIGHTNESS_RATIO or
            bright_ratio_ratio >= LOW_LIGHT_EXIT_BRIGHT_PIXEL_RATIO
        )
        detect_low_light.is_active = not recovered
    else:
        detect_low_light.is_active = low_light_now

    if not detect_low_light.is_active:
        detect_low_light.baseline_brightness = (
            (1.0 - LOW_LIGHT_BASELINE_EMA) * baseline_brightness +
            (LOW_LIGHT_BASELINE_EMA * brightness_mean)
        )
        detect_low_light.baseline_bright_ratio = (
            (1.0 - LOW_LIGHT_RATIO_EMA) * baseline_bright_ratio +
            (LOW_LIGHT_RATIO_EMA * bright_pixel_ratio)
        )

    return detect_low_light.is_active, brightness_mean, bright_pixel_ratio


def chasing_min_area_for_y(norm_y):
    near_weight = clamp((norm_y - 0.20) / 0.70, 0.0, 1.0)
    return CHASE_BACK_FAR_MIN_AREA + ((CHASE_BACK_MIN_AREA - CHASE_BACK_FAR_MIN_AREA) * near_weight)


def police_min_area_for_y(norm_y):
    near_weight = clamp((norm_y - 0.14) / 0.68, 0.0, 1.0)
    return POLICE_FAR_MIN_AREA + ((POLICE_MIN_AREA - POLICE_FAR_MIN_AREA) * near_weight)


def detect_chasing_car(back_frame):
    if not hasattr(detect_chasing_car, "last_result"):
        detect_chasing_car.last_result = None
        detect_chasing_car.last_seen_time = 0.0

    if back_frame is None:
        return None

    height, width = back_frame.shape[:2]
    roi_top = int(height * CHASE_BACK_ROI_TOP)
    roi_bottom = int(height * CHASE_BACK_ROI_BOTTOM)
    roi = back_frame[roi_top:roi_bottom, :]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue_channel = roi[:, :, 0]
    green_channel = roi[:, :, 1]
    red_channel = roi[:, :, 2]

    bgr_mask = (
        (blue_channel >= CHASE_BACK_BLUE_MIN) & (blue_channel <= CHASE_BACK_BLUE_MAX) &
        (green_channel >= CHASE_BACK_GREEN_MIN) & (green_channel <= CHASE_BACK_GREEN_MAX) &
        (red_channel >= CHASE_BACK_RED_MIN) & (red_channel <= CHASE_BACK_RED_MAX) &
        (np.abs(green_channel.astype(np.int16) - blue_channel.astype(np.int16)) <= CHASE_BACK_MAX_BG_DIFF)
    ).astype(np.uint8) * 255
    hsv_mask = cv2.inRange(
        hsv,
        np.array([CHASE_BACK_HUE_MIN, CHASE_BACK_SAT_MIN, CHASE_BACK_VAL_MIN]),
        np.array([CHASE_BACK_HUE_MAX, 255, 255])
    )
    mask = cv2.bitwise_and(bgr_mask, hsv_mask)
    mask = clean_color_mask(mask)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_detection = None
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        center_y = y + (h / 2.0)
        norm_y = center_y / max(float(roi.shape[0]), 1.0)
        if area < chasing_min_area_for_y(norm_y):
            continue

        aspect_ratio = w / max(float(h), 1.0)
        if aspect_ratio < 0.55 or aspect_ratio > 3.20:
            continue
        fill_ratio = area / max(float(w * h), 1.0)
        if fill_ratio < 0.12:
            continue

        center_x = x + (w / 2.0)
        norm_x = normalize_x(center_x, roi.shape[1])
        score = (area * 1.2) + (fill_ratio * 30.0) - (abs(norm_x) * 60.0) - ((1.0 - norm_y) * 8.0)
        detection = {
            'rect': (x, y + roi_top, w, h),
            'norm_x': norm_x,
            'norm_y': norm_y,
            'area': area,
            'near': area >= CHASE_BACK_NEAR_MIN_AREA or norm_y >= CHASE_BACK_NEAR_MIN_Y,
            'same_lane': abs(norm_x) <= CHASE_BACK_CENTER_TOLERANCE,
            'score': score
        }
        if best_detection is None or detection['score'] > best_detection['score']:
            best_detection = detection

    current_time = time.time()
    if best_detection is not None:
        detect_chasing_car.last_result = dict(best_detection)
        detect_chasing_car.last_seen_time = current_time
        return best_detection

    if (
        detect_chasing_car.last_result is not None and
        (current_time - detect_chasing_car.last_seen_time) <= CHASE_BACK_HOLD_TIME
    ):
        return dict(detect_chasing_car.last_result)

    detect_chasing_car.last_result = None
    return None


def detect_police_car(front_frame):
    if not hasattr(detect_police_car, "last_result"):
        detect_police_car.last_result = None
        detect_police_car.last_seen_time = 0.0
        detect_police_car.debug = None

    if front_frame is None:
        detect_police_car.debug = None
        return None

    height, width = front_frame.shape[:2]
    roi_top = int(height * POLICE_ROI_TOP)
    roi_bottom = int(height * POLICE_ROI_BOTTOM)
    roi_left = int(width * POLICE_ROI_LEFT)
    roi_right = int(width * POLICE_ROI_RIGHT)
    roi = front_frame[roi_top:roi_bottom, roi_left:roi_right]
    if roi.size == 0:
        detect_police_car.debug = None
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_bgr_mask = cv2.inRange(roi, POLICE_RED_BGR_MIN, POLICE_RED_BGR_MAX)
    red_hsv_mask = cv2.bitwise_or(
        cv2.inRange(hsv, POLICE_RED_HSV_1_MIN, POLICE_RED_HSV_1_MAX),
        cv2.inRange(hsv, POLICE_RED_HSV_2_MIN, POLICE_RED_HSV_2_MAX)
    )
    blue_bgr_mask = cv2.inRange(roi, POLICE_BLUE_BGR_MIN, POLICE_BLUE_BGR_MAX)
    blue_hsv_mask = cv2.inRange(hsv, POLICE_BLUE_HSV_MIN, POLICE_BLUE_HSV_MAX)

    red_mask = clean_color_mask(cv2.bitwise_and(red_bgr_mask, red_hsv_mask))
    blue_mask = clean_color_mask(cv2.bitwise_and(blue_bgr_mask, blue_hsv_mask))
    red_mask = cv2.dilate(red_mask, np.ones((3, 3), np.uint8), iterations=1)
    blue_mask = cv2.dilate(blue_mask, np.ones((3, 3), np.uint8), iterations=1)

    combined_mask = cv2.bitwise_or(red_mask, blue_mask)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_candidate_boxes = []
    blue_candidate_boxes = []
    best_detection = None
    current_time = time.time()

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)
        norm_x = normalize_x(center_x, roi.shape[1])
        norm_y = center_y / max(float(roi.shape[0]), 1.0)

        if area < police_min_area_for_y(norm_y):
            continue
        if w < POLICE_MIN_BOX_WIDTH or h < POLICE_MIN_BOX_HEIGHT:
            continue
        if w < POLICE_FINAL_MIN_WIDTH or h < POLICE_FINAL_MIN_HEIGHT:
            continue
        if (w / max(float(h), 1.0)) > POLICE_MAX_BOX_ASPECT:
            continue
        if abs(norm_x) > POLICE_CENTER_TOLERANCE:
            continue
        if norm_y < POLICE_MIN_NORM_Y:
            continue

        box_area = max(float(w * h), 1.0)
        red_patch = red_mask[y:y + h, x:x + w]
        blue_patch = blue_mask[y:y + h, x:x + w]
        red_pixels = cv2.countNonZero(red_patch)
        blue_pixels = cv2.countNonZero(blue_patch)
        if red_pixels < POLICE_MIN_COLOR_PIXELS or blue_pixels < POLICE_MIN_COLOR_PIXELS:
            continue
        if (red_pixels / box_area) < POLICE_MIN_COLOR_SHARE:
            continue
        if (blue_pixels / box_area) < POLICE_MIN_COLOR_SHARE:
            continue

        mid_x = w // 2
        left_red = cv2.countNonZero(red_patch[:, :mid_x])
        left_blue = cv2.countNonZero(blue_patch[:, :mid_x])
        right_red = cv2.countNonZero(red_patch[:, mid_x:])
        right_blue = cv2.countNonZero(blue_patch[:, mid_x:])
        half_area = max(float(max(mid_x, 1) * h), 1.0)

        left_red_right_blue = (
            left_red >= POLICE_MIN_HALF_COLOR_PIXELS and
            right_blue >= POLICE_MIN_HALF_COLOR_PIXELS and
            (left_red / half_area) >= POLICE_MIN_HALF_COLOR_SHARE and
            (right_blue / half_area) >= POLICE_MIN_HALF_COLOR_SHARE and
            left_red > left_blue and
            right_blue > right_red
        )
        left_blue_right_red = (
            left_blue >= POLICE_MIN_HALF_COLOR_PIXELS and
            right_red >= POLICE_MIN_HALF_COLOR_PIXELS and
            (left_blue / half_area) >= POLICE_MIN_HALF_COLOR_SHARE and
            (right_red / half_area) >= POLICE_MIN_HALF_COLOR_SHARE and
            left_blue > left_red and
            right_red > right_blue
        )
        if not (left_red_right_blue or left_blue_right_red):
            continue

        pair_fill_ratio = (red_pixels + blue_pixels) / box_area
        if pair_fill_ratio < POLICE_MIN_PAIR_FILL_RATIO:
            continue

        red_contours, _ = cv2.findContours(red_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for sub_contour in red_contours:
            sub_area = cv2.contourArea(sub_contour)
            if sub_area < 6:
                continue
            sx, sy, sw, sh = cv2.boundingRect(sub_contour)
            red_candidate_boxes.append((x + sx + roi_left, y + sy + roi_top, sw, sh))

        blue_contours, _ = cv2.findContours(blue_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for sub_contour in blue_contours:
            sub_area = cv2.contourArea(sub_contour)
            if sub_area < 6:
                continue
            sx, sy, sw, sh = cv2.boundingRect(sub_contour)
            blue_candidate_boxes.append((x + sx + roi_left, y + sy + roi_top, sw, sh))

        detection = {
            'rect': (x + roi_left, y + roi_top, w, h),
            'norm_x': normalize_x(center_x + roi_left, width),
            'norm_y': (center_y + roi_top) / max(float(height), 1.0),
            'norm_w': (2.0 * w) / max(float(width), 1.0),
            'area': float(red_pixels + blue_pixels),
            'near': area >= POLICE_NEAR_MIN_AREA or norm_y >= POLICE_NEAR_MIN_Y,
            'score': (area * 1.10) + (pair_fill_ratio * 24.0) + (norm_y * 16.0) - (abs(norm_x) * 28.0)
        }
        if best_detection is None or detection['score'] > best_detection['score']:
            best_detection = detection

    detect_police_car.debug = {
        'roi_rect': (roi_left, roi_top, roi_right - roi_left, roi_bottom - roi_top),
        'red_boxes': red_candidate_boxes,
        'blue_boxes': blue_candidate_boxes,
        'final_rect': None if best_detection is None else best_detection['rect']
    }

    if best_detection is not None:
        detect_police_car.last_result = dict(best_detection)
        detect_police_car.last_seen_time = current_time
        return best_detection

    if (
        detect_police_car.last_result is not None and
        (current_time - detect_police_car.last_seen_time) <= POLICE_HOLD_TIME
    ):
        return dict(detect_police_car.last_result)

    detect_police_car.last_result = None
    return None


def build_back_debug_frame(back_frame, chasing_car):
    if back_frame is None:
        return None

    debug_back = back_frame.copy()
    height, width = debug_back.shape[:2]
    roi_top = int(height * CHASE_BACK_ROI_TOP)
    roi_bottom = int(height * CHASE_BACK_ROI_BOTTOM)
    center_x = width // 2
    band_half_width = int(width * CHASE_BACK_CENTER_TOLERANCE * 0.5)
    band_left = max(0, center_x - band_half_width)
    band_right = min(width - 1, center_x + band_half_width)

    cv2.rectangle(debug_back, (0, roi_top), (width - 1, roi_bottom), (80, 80, 80), 2)
    cv2.line(debug_back, (center_x, roi_top), (center_x, roi_bottom), (255, 0, 0), 2)
    cv2.rectangle(debug_back, (band_left, roi_top), (band_right, roi_bottom), (255, 255, 0), 2)

    if chasing_car is not None:
        x, y, w, h = chasing_car['rect']
        color = (0, 165, 255) if chasing_car['same_lane'] else (160, 160, 160)
        cv2.rectangle(debug_back, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            debug_back,
            f"ghost {'same lane' if chasing_car['same_lane'] else 'other lane'}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )

    cv2.putText(
        debug_back,
        "Blue=center  Yellow band=same-lane zone",
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    return debug_back


def clean_color_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_tokens(frame, road_mask, roi_top, lane_left_norm, lane_right_norm):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bgr_float = frame.astype(np.float32)
    blue_channel = bgr_float[:, :, 0]
    green_channel = bgr_float[:, :, 1]
    red_channel = bgr_float[:, :, 2]
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, WHITE_MIN_VALUE]),
        np.array([180, WHITE_MAX_SATURATION, 255])
    )
    yellow_core_mask = (
        (red_channel >= YELLOW_CORE_MIN_RED) &
        (green_channel >= YELLOW_CORE_MIN_GREEN) &
        (blue_channel <= YELLOW_CORE_MAX_BLUE) &
        ((red_channel - green_channel) >= YELLOW_CORE_MIN_RG_DIFF) &
        ((red_channel - green_channel) <= YELLOW_CORE_MAX_RG_DIFF) &
        ((green_channel - blue_channel) >= YELLOW_CORE_MIN_GB_DIFF)
    ).astype(np.uint8) * 255
    yellow_bone_mask = cv2.inRange(
        hsv,
        np.array([38, 0, YELLOW_BONE_MIN_VALUE]),
        np.array([58, YELLOW_BONE_MAX_SATURATION, 235])
    )
    yellow_red_exclude_mask = cv2.inRange(
        hsv,
        np.array([YELLOW_RED_EXCLUDE_MIN_H, YELLOW_RED_EXCLUDE_MIN_S, YELLOW_RED_EXCLUDE_MIN_V]),
        np.array([YELLOW_RED_EXCLUDE_MAX_H, 255, 255])
    )
    yellow_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([32, 65, 100]), np.array([56, 255, 255])),
        yellow_core_mask
    )
    yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(yellow_bone_mask))
    yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(white_mask))
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 8, 95]), np.array([14, 190, 255])),
        cv2.inRange(hsv, np.array([166, 8, 95]), np.array([180, 190, 255]))
    )
    red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(white_mask))
    red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(yellow_red_exclude_mask))
    # Remove clearly yellow pixels so yellow tokens do not get double-counted as red.
    red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(yellow_mask))
    masks = {
        'green': cv2.bitwise_and(
            cv2.inRange(hsv, np.array([40, 35, 125]), np.array([75, 150, 255])),
            cv2.bitwise_not(white_mask)
        ),
        'yellow': yellow_mask,
        'gray': cv2.inRange(hsv, np.array([0, 0, GRAY_MIN_VALUE]), np.array([180, GRAY_MAX_SATURATION, GRAY_MAX_VALUE])),
        'red': red_mask
    }

    token_info = {'green': [], 'yellow': [], 'red': [], 'gray': []}
    height, width = frame.shape[:2]
    edge_margin = int(width * EDGE_MARGIN_RATIO)

    for token_type, mask in masks.items():
        mask = clean_color_mask(mask)
        if road_mask is not None:
            roi_mask = np.zeros_like(mask)
            roi_mask[roi_top:, :] = road_mask
            mask = cv2.bitwise_and(mask, roi_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            center_y = y + (h / 2.0)
            norm_y = center_y / height
            if area < min_token_area_for_y(norm_y):
                continue

            if x <= edge_margin or (x + w) >= (width - edge_margin):
                continue

            aspect_ratio = w / max(float(h), 1.0)
            if aspect_ratio < MIN_TOKEN_ASPECT or aspect_ratio > MAX_TOKEN_ASPECT:
                continue

            perimeter = max(cv2.arcLength(contour, True), 1.0)
            compactness = (4.0 * np.pi * area) / (perimeter * perimeter)
            fill_ratio = area / max(float(w * h), 1.0)
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            shifted_contour = contour.copy()
            shifted_contour[:, :, 0] -= x
            shifted_contour[:, :, 1] -= y
            cv2.drawContours(contour_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

            if road_mask is not None:
                y_in_roi = y - roi_top
                if y_in_roi < 0 or (y_in_roi + h) > road_mask.shape[0]:
                    continue

                road_patch = road_mask[y_in_roi:y_in_roi + h, x:x + w]
                if road_patch.shape != contour_mask.shape:
                    continue

                overlap_pixels = cv2.countNonZero(cv2.bitwise_and(contour_mask, road_patch))
                blob_pixels = max(cv2.countNonZero(contour_mask), 1)
                road_overlap = overlap_pixels / blob_pixels
                if road_overlap < MIN_ROAD_OVERLAP:
                    continue

            center_x = x + (w / 2.0)
            norm_x = normalize_x(center_x, width)

            if token_type == 'red':
                # Keep kerb rejection, but allow round red tokens with softer shading/highlights.
                if compactness < RED_MIN_COMPACTNESS:
                    continue
                if fill_ratio > RED_MAX_FILL_RATIO:
                    continue
                min_rect = cv2.minAreaRect(contour)
                min_rect_w, min_rect_h = min_rect[1]
                min_rect_area = max(min_rect_w * min_rect_h, 1.0)
                rotated_extent = area / min_rect_area
                if (
                    abs(norm_x - lane_left_norm) < LANE_BOUNDARY_MARGIN and
                    area > 250
                ) or (
                    abs(norm_x - lane_right_norm) < LANE_BOUNDARY_MARGIN and
                    area > 250
                ):
                    continue
                if rotated_extent > RED_STRIPE_MIN_EXTENT and area > RED_STRIPE_MIN_AREA:
                    continue
                red_patch = red_channel[y:y + h, x:x + w]
                green_patch = green_channel[y:y + h, x:x + w]
                blue_patch = blue_channel[y:y + h, x:x + w]
                hue_patch = hsv[y:y + h, x:x + w, 0]
                sat_patch = hsv[y:y + h, x:x + w, 1]
                mask_pixels = contour_mask > 0
                mean_red = float(np.mean(red_patch[mask_pixels]))
                mean_green = float(np.mean(green_patch[mask_pixels]))
                mean_blue = float(np.mean(blue_patch[mask_pixels]))
                mean_saturation = float(np.mean(sat_patch[mask_pixels]))
                mean_hue = float(np.median(hue_patch[mask_pixels]))
                if mean_red < (mean_green + 12.0) or mean_red < (mean_blue + 8.0):
                    continue
                if mean_saturation > RED_MAX_MEAN_SATURATION:
                    continue
                if not (mean_hue <= RED_MEAN_HUE_MAX or mean_hue >= RED_MEAN_HUE_MIN_WRAP):
                    continue
            elif token_type == 'gray':
                if compactness < GRAY_MIN_COMPACTNESS:
                    continue
                if fill_ratio > GRAY_MAX_FILL_RATIO:
                    continue

            token_info[token_type].append({
                'rect': (x, y, w, h),
                'area': area,
                'norm_x': norm_x,
                'norm_y': norm_y,
                'norm_w': (2.0 * w) / max(float(width), 1.0)
            })
    return token_info


def choose_green_target(tokens, path_center):
    candidate_targets = []
    best_priority_score = None

    for token in tokens['green']:
        if token['norm_y'] < GREEN_MIN_Y:
            continue

        closeness = clamp((token['norm_y'] - GREEN_MIN_Y) / (1.0 - GREEN_MIN_Y), 0.0, 1.0)
        center_score = center_line_overlap(token['norm_x'], path_center)
        wide_alignment = 1.0 - clamp(abs(token['norm_x'] - path_center) / GREEN_MAX_OFFSET, 0.0, 1.0)
        score = (closeness * 1.45) + (center_score * 1.10) + (wide_alignment * 0.45)
        priority_score = token['norm_y'] + (token.get('norm_w', 0.0) * GREEN_SIZE_PRIORITY_WEIGHT)

        candidate_targets.append({
            'token': token,
            'score': score,
            'priority_score': priority_score
        })
        if best_priority_score is None or priority_score > best_priority_score:
            best_priority_score = priority_score

    if not candidate_targets or best_priority_score is None:
        return None

    focused_targets = [
        entry for entry in candidate_targets
        if entry['priority_score'] >= (best_priority_score - GREEN_FRONT_PRIORITY_WINDOW)
    ]
    if not focused_targets:
        return None

    best_entry = max(focused_targets, key=lambda entry: entry['score'])
    best_target = best_entry['token']
    best_score = best_entry['score']

    if best_score < 0.10:
        return None

    return {
        'mode': 'green',
        'target_x': best_target['norm_x'],
        'target_y': best_target['norm_y'],
        'strength': best_score,
        'token': best_target,
        'front_priority_score': best_entry['priority_score'],
        'focused_tokens': [entry['token'] for entry in focused_targets]
    }

def get_lane_index(norm_x, norm_y, road_mask, frame_width, frame_height, roi_top, left_poly=None, right_poly=None):
    """
    Calculates the lane dynamically. Uses Mathematical Polynomials if available
    for perfect curves, otherwise falls back to raw pixel scanning.
    """
    y_px = int(norm_y * frame_height)
    roi_y = y_px - roi_top

    if roi_y < 0 or roi_y >= road_mask.shape[0]:
        return 3

    if left_poly is not None and right_poly is not None:
        left_px = int(left_poly(roi_y))
        right_px = int(right_poly(roi_y))
    else:
        row_pixels = road_mask[roi_y, :]
        occupied_columns = np.where(row_pixels > 0)[0]
        if occupied_columns.size == 0:
            return 3
        left_px = int(occupied_columns[0])
        right_px = int(occupied_columns[-1])

    road_width = right_px - left_px
    if road_width <= 0: return 3

    px_x = ((norm_x + 1.0) * 0.5) * frame_width
    relative_pos = (px_x - left_px) / float(road_width)

    if relative_pos < 0.20: return 1
    elif relative_pos < 0.40: return 2
    elif relative_pos < 0.60: return 3
    elif relative_pos < 0.80: return 4
    else: return 5

def check_edge_lane_escape(tokens, path_center, road_mask, frame_width, frame_height, roi_top, left_poly=None, right_poly=None):
    current_lane = get_lane_index(path_center, 0.95, road_mask, frame_width, frame_height, roi_top, left_poly, right_poly)

    if current_lane not in [1, 5]:
        return None

    for token in tokens['red']:
        if token['norm_y'] < HAZARD_MIN_Y:
            continue

        token_lane = get_lane_index(token['norm_x'], token['norm_y'], road_mask, frame_width, frame_height, roi_top, left_poly, right_poly)

        if token_lane == current_lane:
            if current_lane == 1:
                return {
                    'mode': 'avoid',
                    'target_x': -0.35,
                    'target_y': 0.72,
                    'strength': 1.0,
                    'focused_hazards': [],
                    'gap_centers': [-0.35],
                    'is_edge_escape': True
                }
            elif current_lane == 5:
                return {
                    'mode': 'avoid',
                    'target_x': 0.35,
                    'target_y': 0.72,
                    'strength': 1.0,
                    'focused_hazards': [],
                    'gap_centers': [0.35],
                    'is_edge_escape': True
                }
    return None

def choose_red_target(tokens, path_center):
    candidate_targets = []
    best_priority_score = None

    for token in tokens['red']:
        if token['norm_y'] < HAZARD_MIN_Y:
            continue

        closeness = clamp((token['norm_y'] - HAZARD_MIN_Y) / (1.0 - HAZARD_MIN_Y), 0.0, 1.0)
        center_score = center_line_overlap(token['norm_x'], path_center)
        wide_alignment = 1.0 - clamp(abs(token['norm_x'] - path_center) / GREEN_MAX_OFFSET, 0.0, 1.0)
        score = (closeness * 1.55) + (center_score * 1.10) + (wide_alignment * 0.35)
        priority_score = token['norm_y'] + (token.get('norm_w', 0.0) * HAZARD_SIZE_PRIORITY_WEIGHT)

        candidate_targets.append({
            'token': token,
            'score': score,
            'priority_score': priority_score
        })
        if best_priority_score is None or priority_score > best_priority_score:
            best_priority_score = priority_score

    if not candidate_targets or best_priority_score is None:
        return None

    focused_targets = [
        entry for entry in candidate_targets
        if entry['priority_score'] >= (best_priority_score - HAZARD_FRONT_PRIORITY_WINDOW)
    ]
    if not focused_targets:
        return None

    best_entry = max(focused_targets, key=lambda entry: entry['score'])
    best_target = best_entry['token']
    if best_entry['score'] < 0.10:
        return None

    return {
        'mode': 'police_red',
        'target_x': best_target['norm_x'],
        'target_y': best_target['norm_y'],
        'strength': best_entry['score'],
        'token': best_target,
        'front_priority_score': best_entry['priority_score'],
        'focused_tokens': [entry['token'] for entry in focused_targets]
    }


def choose_police_car_avoidance(police_car, path_center):
    if police_car is None:
        return None

    path_overlap = center_line_overlap(police_car['norm_x'], path_center)
    if path_overlap <= 0.0:
        return None

    closeness = clamp((police_car['norm_y'] - HAZARD_MIN_Y) / (1.0 - HAZARD_MIN_Y), 0.0, 1.0)
    threat = (closeness * 1.45) * path_overlap * 1.45
    interval_half = max((police_car.get('norm_w', 0.14) * 0.60) + HAZARD_CLEARANCE, 0.10)
    left_bound = clamp(police_car['norm_x'] - interval_half, ROAD_LEFT_LIMIT, ROAD_RIGHT_LIMIT)
    right_bound = clamp(police_car['norm_x'] + interval_half, ROAD_LEFT_LIMIT, ROAD_RIGHT_LIMIT)

    left_gap = left_bound - ROAD_LEFT_LIMIT
    right_gap = ROAD_RIGHT_LIMIT - right_bound
    target_x = ROAD_LEFT_LIMIT if left_gap >= right_gap else ROAD_RIGHT_LIMIT

    return {
        'mode': 'police_avoid',
        'target_x': target_x,
        'target_y': max(0.68, police_car['norm_y']),
        'strength': threat,
        'front_priority_score': police_car['norm_y'] + (police_car.get('norm_w', 0.0) * HAZARD_SIZE_PRIORITY_WEIGHT),
        'focused_hazards': [{
            'type': 'police',
            'token': {
                'rect': police_car['rect']
            },
            'threat': threat,
            'left': left_bound,
            'right': right_bound,
            'priority_score': police_car['norm_y']
        }],
        'gap_centers': [target_x]
    }


def choose_hazard_avoidance(tokens, path_center, avoid_red=True):
    blocking_hazards = []
    strongest_threat = 0.0

    hazard_types = ['yellow', 'gray']
    if avoid_red:
        hazard_types.insert(0, 'red')

    for token_type in hazard_types:
        for token in tokens[token_type]:
            if token['norm_y'] < HAZARD_MIN_Y:
                continue

            path_overlap = center_line_overlap(token['norm_x'], path_center)
            if path_overlap <= 0.0:
                continue

            closeness = clamp((token['norm_y'] - HAZARD_MIN_Y) / (1.0 - HAZARD_MIN_Y), 0.0, 1.0)
            if token_type == 'red':
                token_weight = 1.35
            elif token_type == 'yellow':
                token_weight = 1.15
            else:
                token_weight = 1.05
            threat = (closeness * 1.35) * path_overlap * token_weight
            strongest_threat = max(strongest_threat, threat)

            interval_half = max((token.get('norm_w', 0.10) * 0.55) + HAZARD_CLEARANCE, 0.08)
            left_bound = clamp(token['norm_x'] - interval_half, ROAD_LEFT_LIMIT, ROAD_RIGHT_LIMIT)
            right_bound = clamp(token['norm_x'] + interval_half, ROAD_LEFT_LIMIT, ROAD_RIGHT_LIMIT)
            priority_score = token['norm_y'] + (token.get('norm_w', 0.0) * HAZARD_SIZE_PRIORITY_WEIGHT)
            blocking_hazards.append({
                'type': token_type,
                'token': token,
                'threat': threat,
                'left': left_bound,
                'right': right_bound,
                'priority_score': priority_score
            })

    if not blocking_hazards or strongest_threat < 0.18:
        return None

    best_priority_score = max(hazard['priority_score'] for hazard in blocking_hazards)
    focused_hazards = [
        hazard for hazard in blocking_hazards
        if hazard['priority_score'] >= (best_priority_score - HAZARD_FRONT_PRIORITY_WINDOW)
    ]

    if not focused_hazards:
        return None

    strongest_threat = max(hazard['threat'] for hazard in focused_hazards)
    intervals = sorted([(hazard['left'], hazard['right']) for hazard in focused_hazards], key=lambda item: item[0])
    merged_intervals = []
    for left_bound, right_bound in intervals:
        if not merged_intervals or left_bound > merged_intervals[-1][1]:
            merged_intervals.append([left_bound, right_bound])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], right_bound)

    gaps = []
    cursor = ROAD_LEFT_LIMIT
    for left_bound, right_bound in merged_intervals:
        if left_bound > cursor:
            gaps.append((cursor, left_bound))
        cursor = max(cursor, right_bound)
    if cursor < ROAD_RIGHT_LIMIT:
        gaps.append((cursor, ROAD_RIGHT_LIMIT))

    if not gaps:
        leftmost = merged_intervals[0][0]
        rightmost = merged_intervals[-1][1]
        target_x = ROAD_LEFT_LIMIT if abs(leftmost - ROAD_LEFT_LIMIT) >= abs(ROAD_RIGHT_LIMIT - rightmost) else ROAD_RIGHT_LIMIT
        return {
            'mode': 'avoid',
            'target_x': target_x,
            'target_y': 0.72,
            'strength': strongest_threat,
            'focused_hazards': focused_hazards,
            'gap_centers': [target_x]
        }

    candidate_targets = []
    for gap_left, gap_right in gaps:
        gap_width = gap_right - gap_left
        if gap_width <= 0.02:
            continue

        gap_center = (gap_left + gap_right) * 0.5
        alignment = 1.0 - clamp(abs(gap_center - path_center) / max(ROAD_RIGHT_LIMIT - ROAD_LEFT_LIMIT, 1.0), 0.0, 1.0)
        green_bonus = 0.0
        for green_token in tokens['green']:
            if green_token['norm_y'] < GREEN_MIN_Y:
                continue
            if gap_left <= green_token['norm_x'] <= gap_right:
                green_bonus = max(green_bonus, HAZARD_GAP_GREEN_BONUS)

        gap_score = (gap_width * 1.8) + (alignment * 0.9) + green_bonus
        candidate_targets.append((gap_score, gap_center, gap_width))

    if not candidate_targets:
        return None

    candidate_targets.sort(key=lambda entry: entry[0], reverse=True)
    _, target_x, _ = candidate_targets[0]

    return {
        'mode': 'avoid',
        'target_x': target_x,
        'target_y': 0.72,
        'strength': strongest_threat,
        'focused_hazards': focused_hazards,
        'gap_centers': sorted({entry[1] for entry in candidate_targets}),
        'front_priority_score': best_priority_score
    }


def get_stable_path(path_choice, green_still_visible, hazard_still_blocking, police_still_blocking=False):
    current_time = time.time()

    if not hasattr(get_stable_path, "held_path"):
        get_stable_path.held_path = None
        get_stable_path.last_seen_time = 0.0
        get_stable_path.green_centered = False
        get_stable_path.state_label = "released"

    def path_sign(target_x):
        if target_x > 0.0:
            return 1
        if target_x < 0.0:
            return -1
        return 0

    held_path = get_stable_path.held_path

    if (
        held_path is not None and
        held_path.get('mode') == 'green' and
        path_choice is not None and
        path_choice.get('mode') in ('avoid', 'police_avoid') and
        (hazard_still_blocking or police_still_blocking)
    ):
        get_stable_path.held_path = dict(path_choice)
        get_stable_path.last_seen_time = current_time
        get_stable_path.green_centered = False
        get_stable_path.state_label = "locked-avoid"
        return get_stable_path.held_path

    if held_path is not None and held_path.get('mode') == 'avoid' and not hazard_still_blocking:
        get_stable_path.held_path = None
        get_stable_path.green_centered = False
        get_stable_path.state_label = "released"
        held_path = None

    if held_path is not None and held_path.get('mode') == 'police_avoid' and not police_still_blocking:
        get_stable_path.held_path = None
        get_stable_path.green_centered = False
        get_stable_path.state_label = "released"
        held_path = None

    if held_path is not None and held_path.get('mode') == 'green':
        if not get_stable_path.green_centered and abs(held_path['target_x']) <= GREEN_HOLD_CENTER_BAND:
            get_stable_path.green_centered = True

        if get_stable_path.green_centered:
            if green_still_visible:
                get_stable_path.state_label = "locked-green-hold"
                if path_choice is not None and path_choice.get('mode') == 'green':
                    get_stable_path.held_path = dict(path_choice)
                    get_stable_path.last_seen_time = current_time
                    held_path = get_stable_path.held_path
                return held_path

            get_stable_path.held_path = None
            get_stable_path.green_centered = False
            get_stable_path.state_label = "released"
            held_path = None

    if held_path is None:
        if path_choice is not None:
            get_stable_path.held_path = dict(path_choice)
            get_stable_path.last_seen_time = current_time
            get_stable_path.green_centered = False
            get_stable_path.state_label = "locked-green-move" if path_choice.get('mode') == 'green' else "locked-avoid"
            return get_stable_path.held_path
        get_stable_path.state_label = "released"
        return None

    if path_choice is not None:
        same_mode = path_choice.get('mode') == held_path.get('mode')
        same_side = path_sign(path_choice['target_x']) == path_sign(held_path['target_x'])
        if same_mode and same_side:
            get_stable_path.held_path = dict(path_choice)
            get_stable_path.last_seen_time = current_time
            if get_stable_path.held_path.get('mode') == 'green':
                if not get_stable_path.green_centered and abs(get_stable_path.held_path['target_x']) <= GREEN_HOLD_CENTER_BAND:
                    get_stable_path.green_centered = True
                get_stable_path.state_label = "locked-green-hold" if get_stable_path.green_centered else "locked-green-move"
            else:
                get_stable_path.state_label = "locked-avoid"
            return get_stable_path.held_path
        if held_path.get('mode') == 'green':
            get_stable_path.state_label = "locked-green-hold" if get_stable_path.green_centered else "locked-green-move"
        else:
            get_stable_path.state_label = "locked-avoid"
        return held_path

    if (current_time - get_stable_path.last_seen_time) <= PATH_LOST_TIMEOUT:
        if held_path.get('mode') == 'green':
            get_stable_path.state_label = "locked-green-hold" if get_stable_path.green_centered else "locked-green-move"
        else:
            get_stable_path.state_label = "locked-avoid"
        return held_path

    get_stable_path.held_path = None
    get_stable_path.green_centered = False
    get_stable_path.state_label = "released"
    return None


def get_road_bounds_at_target_row(road_mask, target_y_norm):
    if road_mask is None or road_mask.size == 0:
        return None

    height, width = road_mask.shape[:2]
    row_index = int(clamp(target_y_norm * height, 0, height - 1))
    band_radius = max(1, ROAD_BOUNDARY_ROW_BAND)
    row_start = max(0, row_index - band_radius)
    row_end = min(height, row_index + band_radius + 1)
    band = road_mask[row_start:row_end, :]

    occupied_columns = np.where(np.any(band > 0, axis=0))[0]
    if occupied_columns.size == 0:
        return None

    left_bound = int(occupied_columns[0]) + ROAD_BOUNDARY_MARGIN_PIXELS
    right_bound = int(occupied_columns[-1]) - ROAD_BOUNDARY_MARGIN_PIXELS
    if left_bound >= right_bound:
        return None

    return left_bound, right_bound


def apply_boundary_fallback(target_choice, road_mask, frame_width):
    if target_choice is None or target_choice.get('mode') != 'avoid':
        return target_choice

    road_bounds = get_road_bounds_at_target_row(road_mask, target_choice.get('target_y', 0.72))
    if road_bounds is None:
        return target_choice

    left_bound, right_bound = road_bounds

    def target_inside_bounds(target_x_norm):
        target_x_px = int(((target_x_norm + 1.0) * 0.5) * frame_width)
        return left_bound <= target_x_px <= right_bound

    if target_inside_bounds(target_choice['target_x']):
        return target_choice

    alternate_centers = [
        center for center in target_choice.get('gap_centers', [])
        if target_inside_bounds(center)
    ]

    if not alternate_centers:
        return None

    replacement = dict(target_choice)
    replacement['target_x'] = min(alternate_centers, key=lambda center: abs(center))
    return replacement


def analyse_drive(front_frame, back_frame=None):
    if not hasattr(analyse_drive, "chase_override_active"):
        analyse_drive.chase_override_active = False
    if not hasattr(analyse_drive, "police_event_until"):
        analyse_drive.police_event_until = 0.0
    if not hasattr(analyse_drive, "police_red_centered"):
        analyse_drive.police_red_centered = False
    if not hasattr(analyse_drive, "police_red_missing_since"):
        analyse_drive.police_red_missing_since = None

    height, width = front_frame.shape[:2]
    current_time = time.time()
    low_light_active, scene_brightness, bright_pixel_ratio = detect_low_light(front_frame)
    chasing_car = detect_chasing_car(back_frame)
    police_car = None if low_light_active else detect_police_car(front_frame)
    if police_car is not None:
        analyse_drive.police_event_until = max(analyse_drive.police_event_until, current_time + POLICE_EVENT_DURATION)
    police_active = current_time < analyse_drive.police_event_until
    police_time_left = max(0.0, analyse_drive.police_event_until - current_time)
    if chasing_car is not None and chasing_car.get('near', False):
        analyse_drive.chase_override_active = True
    elif chasing_car is None:
        analyse_drive.chase_override_active = False

    chase_visible = chasing_car is not None
    chasing_same_lane = chase_visible and chasing_car['same_lane']
    chase_override_active = analyse_drive.chase_override_active and chase_visible
    roi_top = int(height * ROI_START)
    roi = front_frame[roi_top:, :]
    roi_height, roi_width = roi.shape[:2]

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_track = np.array([0, 0, 35])
    upper_track = np.array([180, 40, 120])

    path_mask = cv2.inRange(hsv_roi, lower_track, upper_track)

    kernel = np.ones((7, 7), np.uint8)
    path_mask = cv2.morphologyEx(path_mask, cv2.MORPH_CLOSE, kernel)
    path_mask = cv2.dilate(path_mask, kernel, iterations=2)

    road_mask = build_road_mask(path_mask)

    center_x = roi_width // 2
    path_center = normalize_x(center_x, roi_width)
    lane_left_norm = -0.55
    lane_right_norm = 0.55
    left_poly = None
    right_poly = None
    with data_lock:
        sent_steering = shared_data['sent_steering_input']
        sent_acceleration = shared_data['sent_acceleration_input']
    if low_light_active:
        tokens = {'green': [], 'yellow': [], 'red': [], 'gray': []}
        green_choice = None
        red_choice = None
        hazard_choice = None
        police_avoid_choice = None
        edge_escape_choice = None
    else:
        tokens = find_tokens(front_frame, road_mask, roi_top, lane_left_norm, lane_right_norm)
        red_choice = choose_red_target(tokens, path_center) if police_active else None
        police_avoid_choice = choose_police_car_avoidance(police_car, path_center) if police_active else None

        points_y = []
        left_points_x = []
        right_points_x = []

        for current_y in range(roi_height - 1, 0, -5):
            row_pixels = road_mask[current_y, :]
            occupied = np.where(row_pixels > 0)[0]
            if occupied.size > 0:
                points_y.append(current_y)
                left_points_x.append(occupied[0])
                right_points_x.append(occupied[-1])

        if len(points_y) > 10:
            left_poly = np.poly1d(np.polyfit(points_y, left_points_x, 2))
            right_poly = np.poly1d(np.polyfit(points_y, right_points_x, 2))

        edge_escape_choice = check_edge_lane_escape(tokens, path_center, road_mask, width, height, roi_top, left_poly, right_poly)

        green_choice = choose_green_target(tokens, path_center)
        hazard_choice = choose_hazard_avoidance(tokens, path_center)

    green_front_priority = green_choice.get('front_priority_score', -1.0) if green_choice is not None else -1.0
    red_front_priority = red_choice.get('front_priority_score', -1.0) if red_choice is not None else -1.0
    hazard_front_priority = hazard_choice.get('front_priority_score', -1.0) if hazard_choice is not None else -1.0
    police_front_priority = police_avoid_choice.get('front_priority_score', -1.0) if police_avoid_choice is not None else -1.0

    if police_avoid_choice is not None:
        raw_path_choice = police_avoid_choice
    elif chase_visible:
        mirrored_chase_x = -chasing_car['norm_x']
        chase_direction = -1.0 if mirrored_chase_x >= 0.0 else 1.0
        raw_path_choice = {
            'mode': 'chase_avoid',
            'target_x': (0.72 if chase_override_active else 0.52) * chase_direction,
            'target_y': 0.76,
            'strength': 1.0 if chase_override_active else 0.82
        }
    elif police_active and red_choice is not None and red_front_priority >= green_front_priority:
        raw_path_choice = red_choice
    elif edge_escape_choice is not None:
        raw_path_choice = edge_escape_choice
    elif (
        hazard_choice is not None and
        hazard_choice['strength'] >= HAZARD_OVERRIDE_THREAT and
        hazard_front_priority >= (green_front_priority + GREEN_FRONT_PRIORITY_ADVANTAGE)
    ):
        raw_path_choice = hazard_choice
    elif green_choice is not None:
        raw_path_choice = green_choice
    elif hazard_choice is not None and hazard_choice['strength'] >= HAZARD_OVERRIDE_THREAT:
        raw_path_choice = hazard_choice
    else:
        raw_path_choice = None

    if chase_visible:
        target_choice = raw_path_choice
    else:
        target_choice = get_stable_path(
            raw_path_choice,
            (green_choice is not None) or (red_choice is not None),
            hazard_choice is not None,
            police_avoid_choice is not None
        )
        target_choice = apply_boundary_fallback(target_choice, road_mask, width)
    if police_avoid_choice is not None and raw_path_choice is police_avoid_choice:
        focused_hazards = police_avoid_choice['focused_hazards']
    else:
        focused_hazards = hazard_choice['focused_hazards'] if hazard_choice is not None else []
    focused_greens = green_choice['focused_tokens'] if green_choice is not None else []
    focused_reds = red_choice['focused_tokens'] if red_choice is not None else []
    path_lock_state = (
        "chase-override" if chase_override_active else
        ("chase-track" if chase_visible else getattr(get_stable_path, "state_label", "released"))
    )

    drive_mode = 'path'
    steering = 0.0
    green_hold_aligned = False

    if target_choice is not None:
        if target_choice.get('mode') == 'green' and abs(target_choice['target_x']) <= GREEN_HOLD_CENTER_BAND:
            steering = 0.0
            green_hold_aligned = True
        elif target_choice['target_x'] > 0.0:
            steering = 1.0
        elif target_choice['target_x'] < 0.0:
            steering = -1.0
        else:
            steering = 0.0
        drive_mode = target_choice['mode']

    if police_active:
        if red_choice is not None:
            analyse_drive.police_red_missing_since = None
        if drive_mode == 'police_red' and target_choice is not None and abs(target_choice['target_x']) <= PATH_RELEASE_CENTER_THRESHOLD:
            analyse_drive.police_red_centered = True
        if (
            analyse_drive.police_red_centered and
            red_choice is None and
            police_avoid_choice is None
        ):
            if analyse_drive.police_red_missing_since is None:
                analyse_drive.police_red_missing_since = current_time
            elif (current_time - analyse_drive.police_red_missing_since) >= POLICE_RED_CLEAR_CONFIRM_TIME:
                analyse_drive.police_event_until = current_time
                police_active = False
                police_time_left = 0.0
                analyse_drive.police_red_centered = False
                analyse_drive.police_red_missing_since = None
        elif red_choice is not None:
            analyse_drive.police_red_missing_since = None
    else:
        analyse_drive.police_red_centered = False
        analyse_drive.police_red_missing_since = None

    acceleration = TEST_THROTTLE
    if low_light_active:
        acceleration = -1.0
        drive_mode = 'low_light'

    debug_frame = front_frame.copy()
    back_debug_frame = build_back_debug_frame(back_frame, chasing_car)
    cv2.rectangle(debug_frame, (0, roi_top), (width - 1, height - 1), (80, 80, 80), 2)
    cv2.line(debug_frame, (center_x, roi_top), (center_x, height - 1), (255, 0, 0), 2)
    green_band_half_width = int(width * GREEN_HOLD_CENTER_BAND * 0.5)
    green_band_left = max(0, center_x - green_band_half_width)
    green_band_right = min(width - 1, center_x + green_band_half_width)
    cv2.rectangle(debug_frame, (green_band_left, roi_top), (green_band_right, height - 1), (0, 165, 255), 2)

    hsv_debug = cv2.cvtColor(front_frame, cv2.COLOR_BGR2HSV)

    lower_grey = np.array([0, 0, 85])
    upper_grey = np.array([180, 40, 110])
    lane_mask_debug = cv2.inRange(hsv_debug, lower_grey, upper_grey)

    lane_on_road = np.zeros_like(lane_mask_debug)
    lane_on_road[roi_top:, :] = cv2.bitwise_and(lane_mask_debug[roi_top:, :], road_mask)
    debug_frame[lane_on_road > 0] = [255, 255, 0]

    if left_poly is not None and right_poly is not None:
        prev_dividers = None
        for roi_y in range(roi_height - 1, 0, -15):
            fit_left_x = left_poly(roi_y)
            fit_right_x = right_poly(roi_y)
            road_width = fit_right_x - fit_left_x

            if road_width > 20:
                current_dividers = [
                    int(fit_left_x + road_width * 0.20),
                    int(fit_left_x + road_width * 0.40),
                    int(fit_left_x + road_width * 0.60),
                    int(fit_left_x + road_width * 0.80)
                ]

                screen_y = roi_top + roi_y

                if prev_dividers:
                    for i in range(4):
                        cv2.line(debug_frame, (prev_dividers[i], screen_y + 15), (current_dividers[i], screen_y), (200, 200, 200), 1)
                prev_dividers = current_dividers

    road_overlay = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
    road_overlay[:, :, 0] = 0
    road_overlay[:, :, 2] = 0
    debug_frame[roi_top:, :] = cv2.addWeighted(debug_frame[roi_top:, :], 1.0, road_overlay, 0.18, 0)

    for token_type, color in (('green', (0, 255, 0)), ('yellow', (0, 255, 255)), ('red', (0, 0, 255)), ('gray', (180, 180, 180))):
        for token in tokens[token_type]:
            x, y, w, h = token['rect']
            cv2.rectangle(
                debug_frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )
            cv2.putText(
                debug_frame,
                token_type,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA
            )

    if focused_hazards:
        front_band_y = min(hazard['token']['rect'][1] for hazard in focused_hazards)
        cv2.line(debug_frame, (0, front_band_y), (width - 1, front_band_y), (255, 255, 0), 2)
        cv2.putText(
            debug_frame,
            "front priority",
            (10, max(18, front_band_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )
        for hazard in focused_hazards:
            x, y, w, h = hazard['token']['rect']
            cv2.rectangle(debug_frame, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 255, 0), 2)

    if focused_greens:
        green_front_band_y = min(token['rect'][1] for token in focused_greens)
        cv2.line(debug_frame, (0, green_front_band_y), (width - 1, green_front_band_y), (0, 255, 0), 2)
        cv2.putText(
            debug_frame,
            "green front priority",
            (10, max(18, green_front_band_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        for token in focused_greens:
            x, y, w, h = token['rect']
            cv2.rectangle(debug_frame, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 2)

    if green_hold_aligned:
        cv2.putText(
            debug_frame,
            "green aligned -> hold straight",
            (10, 148),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    if focused_reds:
        red_front_band_y = min(token['rect'][1] for token in focused_reds)
        cv2.line(debug_frame, (0, red_front_band_y), (width - 1, red_front_band_y), (255, 0, 255), 2)
        cv2.putText(
            debug_frame,
            "red front priority",
            (10, max(18, red_front_band_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )
        for token in focused_reds:
            x, y, w, h = token['rect']
            cv2.rectangle(debug_frame, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 0, 255), 2)

    if target_choice is not None:
        target_x = int(((target_choice['target_x'] + 1.0) * 0.5) * width)
        target_color = (0, 255, 0) if target_choice['mode'] != 'police_red' else (0, 0, 255)
        cv2.line(debug_frame, (width // 2, height - 10), (target_x, int(target_choice['target_y'] * height)), target_color, 2)
        cv2.putText(
            debug_frame,
            f"target={target_choice['mode']}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            target_color,
            2,
            cv2.LINE_AA
        )

    cv2.putText(
        debug_frame,
        f"mode={drive_mode} steer={sent_steering:+.2f} accel={sent_acceleration:.2f} path={path_lock_state}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    if edge_escape_choice is not None:
        cv2.putText(
            debug_frame,
            "!!! EMERGENCY EDGE ESCAPE !!!",
            (width // 2 - 140, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    if low_light_active:
        cv2.putText(
            debug_frame,
            f"LOW LIGHT RECOVERY brightness={scene_brightness:.1f} bright_ratio={bright_pixel_ratio:.2f}",
            (10, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA
        )

    if police_active:
        cv2.putText(
            debug_frame,
            f"POLICE EVENT active {police_time_left:.1f}s",
            (10, 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

    if police_car is not None:
        x, y, w, h = police_car['rect']
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(
            debug_frame,
            f"police {'avoid' if drive_mode == 'police_avoid' else 'active'}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

    police_debug = getattr(detect_police_car, "debug", None)
    if police_debug is not None:
        rx, ry, rw, rh = police_debug['roi_rect']
        cv2.rectangle(debug_frame, (rx, ry), (rx + rw, ry + rh), (180, 180, 180), 1)
        for bx, by, bw, bh in police_debug['red_boxes']:
            cv2.rectangle(debug_frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)
        for bx, by, bw, bh in police_debug['blue_boxes']:
            cv2.rectangle(debug_frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 1)
        cv2.putText(
            debug_frame,
            "police roi / red / blue",
            (rx, max(18, ry - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

    if chasing_car is not None:
        x, y, w, h = chasing_car['rect']
        chase_color = (0, 165, 255) if chase_override_active else ((0, 220, 220) if chasing_same_lane else (160, 160, 160))
        box_w = max(24, int(width * 0.10))
        box_h = max(16, int(height * 0.06))
        mirrored_norm_x = -chasing_car['norm_x']
        front_box_x = int(clamp(((mirrored_norm_x + 1.0) * 0.5 * width) - (box_w * 0.5), 0, width - box_w - 1))
        front_box_y = 112
        cv2.putText(
            debug_frame,
            f"CHASE BACK {'OVERRIDE' if chase_override_active else ('BLOCKING' if chasing_same_lane else 'seen')} x={chasing_car['norm_x']:+.2f}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            chase_color,
            2,
            cv2.LINE_AA
        )
        rear_center_x = int(((mirrored_norm_x + 1.0) * 0.5) * width)
        cv2.line(debug_frame, (rear_center_x, 0), (rear_center_x, 40), chase_color, 2)
        cv2.rectangle(debug_frame, (front_box_x, front_box_y), (front_box_x + box_w, front_box_y + box_h), chase_color, 2)
        cv2.putText(
            debug_frame,
            "ghost rear",
            (front_box_x, min(height - 10, front_box_y + box_h + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            chase_color,
            2,
            cv2.LINE_AA
        )

    cv2.putText(
        debug_frame,
        "Blue=center  Orange band=green hold zone  Green path=target  Green/Yellow/Red/Gray=detected objects",
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    is_edge_escape = target_choice.get('is_edge_escape', False) if target_choice is not None else False

    return steering, acceleration, drive_mode, debug_frame, back_debug_frame,is_edge_escape


def processing_task():
    global is_running
    global last_status_print

    if not hasattr(processing_task, "last_escape_print"):
        processing_task.last_escape_print = 0.0

    if quit_requested():
        is_running = False
        return

    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame = shared_data['latest_back_frame']

    if front_frame is not None:
        steering, acceleration, drive_mode, debug_frame, back_debug_frame, is_edge_escape = analyse_drive(front_frame, back_frame)

        current_time = time.time()
        if is_edge_escape and (current_time - processing_task.last_escape_print > 1.0):
            print("\n[ALERT] !!! EMERGENCY EDGE ESCAPE TRIGGERED !!!\n")
            processing_task.last_escape_print = current_time

        with data_lock:
            shared_data['steering_input'] = steering
            shared_data['acceleration_input'] = acceleration
            shared_data['drive_mode'] = drive_mode
            shared_data['debug_frame'] = debug_frame
            shared_data['back_debug_frame'] = back_debug_frame

        current_time = time.time()
        if current_time - last_status_print >= 1.0:
            with data_lock:
                print(
                    "AutoDrive | "
                    f"mode={shared_data['drive_mode']} "
                    f"steering={shared_data['steering_input']:+.2f} "
                    f"accel={shared_data['acceleration_input']:.2f}"
                )
            last_status_print = current_time
    else:
        with data_lock:
            shared_data['steering_input'] = 0.0
            shared_data['acceleration_input'] = TEST_THROTTLE
            shared_data['drive_mode'] = 'search'
            shared_data['back_debug_frame'] = None


def send_controls_task():
    global control_conn
    if control_conn is None:
        return

    if not hasattr(send_controls_task, "last_tap_time"):
        send_controls_task.last_tap_time = 0.0
        send_controls_task.last_requested_direction = 0
        send_controls_task.direction_streak = 0
        send_controls_task.release_pending = False
        send_controls_task.last_pulse_direction = 0

    with data_lock:
        front_frame = shared_data['latest_front_frame']
        desired_steering = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']
        drive_mode = shared_data['drive_mode']

    current_time = time.time()

    if front_frame is None:
        steering_input = 0.0
        acceleration_input = TEST_THROTTLE
        send_controls_task.release_pending = False
        send_controls_task.last_pulse_direction = 0
    else:
        steering_input = 0.0
        tap_cooldown = GREEN_STEERING_TAP_COOLDOWN if drive_mode == 'green' else STEERING_TAP_COOLDOWN
        desired_direction = 0
        if desired_steering > 0.0:
            desired_direction = 1
        elif desired_steering < 0.0:
            desired_direction = -1

        if desired_direction == 0:
            send_controls_task.last_requested_direction = 0
            send_controls_task.direction_streak = 0
            send_controls_task.release_pending = False
            send_controls_task.last_pulse_direction = 0
        elif desired_direction == send_controls_task.last_requested_direction:
            send_controls_task.direction_streak += 1
        else:
            send_controls_task.last_requested_direction = desired_direction
            send_controls_task.direction_streak = 1
            send_controls_task.release_pending = False

        if send_controls_task.release_pending:
            steering_input = 0.0
            send_controls_task.release_pending = False
        elif (
            desired_direction != 0 and
            send_controls_task.direction_streak >= STEERING_CONFIRM_CYCLES and
            (current_time - send_controls_task.last_tap_time) >= tap_cooldown
        ):
            steering_input = float(desired_direction)
            send_controls_task.last_tap_time = current_time
            send_controls_task.last_pulse_direction = desired_direction
            send_controls_task.release_pending = True

    with data_lock:
        shared_data['sent_steering_input'] = steering_input
        shared_data['sent_acceleration_input'] = acceleration_input

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        control_conn = None


# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")

    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()

    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")

    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.LOW, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.005, priority=TaskPriority.HIGH, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)

    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()

    try:
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()

    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
