import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes
import queue

CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

front_frame_queue = queue.LifoQueue(maxsize=1)
back_frame_queue = queue.LifoQueue(maxsize=1)
command_queue = queue.LifoQueue(maxsize=1)

is_running = True

steer_state = {
    'status': 'IDLE',          
    'tap_duration': 0.25,     
    'cooldown_duration': 0.05, 
    'end_time': 0.0,
    'active_steer': 0.0
}

class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
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
        except Exception as e:
            pass

        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)


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

def read_single_camera(sock, window_name, target_queue):
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
                if target_queue.full():
                    try: target_queue.get_nowait()
                    except queue.Empty: pass
                target_queue.put(frame, block=False)
                
                # frame_resized = cv2.resize(frame, (640, 480))
                # cv2.imshow(window_name, frame_resized)
                # cv2.waitKey(1)
                
    except Exception as e:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", front_frame_queue)

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", back_frame_queue)

def processing_task():
    front_frame = None
    back_frame = None
    
    try: front_frame = front_frame_queue.get_nowait()
    except queue.Empty: pass
    
    try: back_frame = back_frame_queue.get_nowait()
    except queue.Empty: pass
    
    acceleration_input = 1.0 
    rear_danger = False      

    if back_frame is not None:
        hsv_back = cv2.cvtColor(back_frame, cv2.COLOR_BGR2HSV)
        lower_car = np.array([0, 0, 0])
        upper_car = np.array([180, 50, 80]) 
        
        mask_car = cv2.inRange(hsv_back, lower_car, upper_car)
        car_contours, _ = cv2.findContours(mask_car, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if car_contours:
            largest_car = max(car_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_car) > 15000: 
                rear_danger = True

    action_triggered = False
    target_x_offset = 0.0
    
    closest_green_cnt = None
    closest_danger_cnt = None
    roi_top = 0

    if front_frame is not None:
        height, width = front_frame.shape[:2]
        center_x = width // 2
        
        roi_top = int(height * 0.30)
        roi_frame = front_frame[roi_top:height, 0:width].copy()
        roi_height = roi_frame.shape[0]

        bottom_left = [0, roi_height]                  
        top_left = [int(width * 0.15), 0]              
        top_right = [int(width * 0.85), 0]             
        bottom_right = [width, roi_height]
        
        road_polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)
        
        mask = np.zeros_like(roi_frame)
        cv2.fillPoly(mask, road_polygon, (255, 255, 255))
        roi_frame = cv2.bitwise_and(roi_frame, mask)
        
        hsv_front = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        
        lower_grass = np.array([50, 80, 20])
        upper_grass = np.array([75, 150, 80])
        

        bad_grass_mask = cv2.inRange(hsv_front, lower_grass, upper_grass)
        
        good_vision_mask = cv2.bitwise_not(bad_grass_mask)
        
        hsv_front = cv2.bitwise_and(hsv_front, hsv_front, mask=good_vision_mask)
       

        lower_green = np.array([35, 50, 50]) 
        upper_green = np.array([85, 255, 255])
        
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        
        raw_mask_green = cv2.inRange(hsv_front, lower_green, upper_green)
        raw_mask_red = cv2.bitwise_or(cv2.inRange(hsv_front, lower_red1, upper_red1), 
                                      cv2.inRange(hsv_front, lower_red2, upper_red2))
        raw_mask_yellow = cv2.inRange(hsv_front, lower_yellow, upper_yellow)
        
        raw_mask_danger = cv2.bitwise_or(raw_mask_red, raw_mask_yellow)
        
        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(raw_mask_green, cv2.MORPH_OPEN, kernel)
        mask_danger = cv2.morphologyEx(raw_mask_danger, cv2.MORPH_OPEN, kernel)
        
        danger_contours, _ = cv2.findContours(mask_danger, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        min_token_area = 100
        max_token_area = 150000 
        

        def get_closest_contour(contours):
            closest_cnt = None
            max_bottom_y = -1
            valid_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_token_area < area < max_token_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    aspect_ratio = float(w) / float(h)
                    
                    rect_area = w * h
                    extent = float(area) / rect_area
                    

                    if (0.7 <= aspect_ratio <= 1.3) and (0.6 <= extent <= 0.9):
                        bottom_y = y + h 
                        if bottom_y > max_bottom_y:
                            max_bottom_y = bottom_y
                            closest_cnt = cnt
                            valid_area = area
                            
            return closest_cnt, valid_area, max_bottom_y

        closest_danger_cnt, danger_area, danger_y = get_closest_contour(danger_contours)
        closest_green_cnt, green_area, green_y = get_closest_contour(green_contours)

        left_lane_edge = width * 0.40
        right_lane_edge = width * 0.60
        
        danger_lane = None
        danger_center_x = 0
        if closest_danger_cnt is not None:
            dx, dy, dw, dh = cv2.boundingRect(closest_danger_cnt)
            danger_center_x = dx + dw//2
            if danger_center_x < left_lane_edge: danger_lane = "LEFT"
            elif danger_center_x > right_lane_edge: danger_lane = "RIGHT"
            else: danger_lane = "CENTER"

        green_lane = None
        green_center_x = 0
        if closest_green_cnt is not None:
            gx, gy, gw, gh = cv2.boundingRect(closest_green_cnt)
            green_center_x = gx + gw//2
            if green_center_x < left_lane_edge: green_lane = "LEFT"
            elif green_center_x > right_lane_edge: green_lane = "RIGHT"
            else: green_lane = "CENTER"

        action_triggered = False
        target_x_offset = 0.0
        
        
        if closest_danger_cnt is not None:
            action_triggered = True
           
            if closest_green_cnt is not None and (green_y > danger_y + 30) and (green_lane != danger_lane):

                pixel_offset = green_center_x - center_x
                target_x_offset = pixel_offset / (width // 2)
            else:
                if danger_lane == "LEFT":
                    target_pixel = width * 0.75 
                elif danger_lane == "RIGHT":
                    target_pixel = width * 0.25 
                else: 
                    
                    target_pixel = width * 0.80 if danger_center_x < center_x else width * 0.20
                    
                pixel_offset = target_pixel - center_x
                target_x_offset = (pixel_offset / (width // 2)) * 1.2
                
        elif closest_green_cnt is not None:
            pixel_offset = green_center_x - center_x
            target_x_offset = pixel_offset / (width // 2)
            action_triggered = True

    final_steering_input = 0.0
    acceleration_input = 1.0 
    
    if action_triggered:

        sensitivity = 8.5
        
        raw_steer = target_x_offset * sensitivity
        final_steering_input = max(min(raw_steer, 1.0), -1.0)
        
        if abs(final_steering_input) < 0.02:
            final_steering_input = 0.0

        turn_intensity = abs(final_steering_input)
        
        min_speed = 0.7       
        brake_strength = 0.3  
        
        if turn_intensity < 0.4:
            acceleration_input = 1.0
        else:
            calculated_speed = 1.0 - (turn_intensity * brake_strength)
            acceleration_input = max(calculated_speed, min_speed)

    if front_frame is not None:
        if closest_green_cnt is not None:
            gx, gy, gw, gh = cv2.boundingRect(closest_green_cnt)
            cv2.rectangle(front_frame, (gx, gy + roi_top), (gx + gw, gy + gh + roi_top), (0, 255, 0), 2)
            
        if closest_danger_cnt is not None:
            dx, dy, dw, dh = cv2.boundingRect(closest_danger_cnt)
            cv2.rectangle(front_frame, (dx, dy + roi_top), (dx + dw, dy + dh + roi_top), (0, 0, 255), 2)

        height, width = front_frame.shape[:2]
        center_x = width // 2
        
        cv2.line(front_frame, (center_x, height), (center_x, roi_top), (255, 0, 0), 2)
        steer_pixel_x = int(center_x + (final_steering_input * (width // 2)))
        line_color = (0, 255, 0) if acceleration_input >= 0.9 else (0, 165, 255) 
        cv2.line(front_frame, (center_x, height), (steer_pixel_x, roi_top), line_color, 4)

        if 'road_polygon' in locals() and road_polygon is not None:
            hud_polygon = road_polygon + [0, roi_top]
            cv2.polylines(front_frame, [hud_polygon], isClosed=True, color=(255, 0, 255), thickness=2)

        cv2.putText(front_frame, f"Steer: {final_steering_input:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(front_frame, f"Gas: {acceleration_input:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Autopilot HUD", cv2.resize(front_frame, (640, 480)))
        
    if back_frame is not None:
        cv2.imshow("Rear View", cv2.resize(back_frame, (640, 480)))

    cv2.waitKey(1)
        
    if command_queue.full():
        try: command_queue.get_nowait()
        except queue.Empty: pass
    command_queue.put((final_steering_input, acceleration_input), block=False)


def send_controls_task():
    global control_conn
    if control_conn is None:
        return
    
    try:
        steering_input, acceleration_input = command_queue.get_nowait()
    except queue.Empty:
        steering_input = 0.0
        acceleration_input = 1.0

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        control_conn = None


if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")
    
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
    
    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")
    
    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    
    t_processing = RTTask("Processing", period=0.010, priority=TaskPriority.MEDIUM, execute_func=processing_task)
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
    
    if front_camera_sock: front_camera_sock.close()
    if back_camera_sock: back_camera_sock.close()
    if control_conn: control_conn.close()
    
    cv2.destroyAllWindows()
    print("System terminated cleanly.")