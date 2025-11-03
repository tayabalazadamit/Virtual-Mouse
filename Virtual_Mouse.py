import cv2
import numpy as np
import mediapipe as mp
import math, time, ctypes
import pyautogui
from screeninfo import get_monitors
from collections import deque


try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

pyautogui.FAILSAFE = False


screen = get_monitors()[0]
SCREEN_W, SCREEN_H = screen.width, screen.height


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def smooth_point(prev, new, alpha=0.25):
    if prev is None:
        return new
    return ((1-alpha)*prev[0] + alpha*new[0], (1-alpha)*prev[1] + alpha*new[1])

def lm_xy(landmark, w, h, mirror=True):
    x = landmark.x
    if mirror:
        x = 1.0 - x
    return (x*w, landmark.y*h)

def map_to_screen(px, py, cam_w, cam_h):
    sx = int(np.clip(px / cam_w * SCREEN_W, 0, SCREEN_W - 1))
    sy = int(np.clip(py / cam_h * SCREEN_H, 0, SCREEN_H - 1))
    return sx, sy


PINCH_RATIO_THRESHOLD = 0.35   
RIGHT_PINCH_RATIO     = 0.35   
QUICK_PINCH_MS        = 200    

TWO_FINGER_RATIO_THRESHOLD = 0.40  
SCROLL_GAIN                = 0.6   
SCROLL_DEBOUNCE_FRAMES     = 4     


cap = cv2.VideoCapture(0)
smoothed_cursor = None
cursor_trail = deque(maxlen=8)


is_left_dragging = False
left_pinch_active = False
left_pinch_start_ms = 0


right_pinch_active = False
right_click_armed = True


scroll_mode = False
prev_scroll_y = None
scroll_gap_frames = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

   
    index_tip_px = None
    left_pinch_now = False
    right_pinch_now = False
    two_fingers_now = False
    pinch_ratio_idx = 0.0
    pinch_ratio_mid = 0.0
    two_finger_ratio = 1.0

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

     
        thumb_tip   = lm_xy(lm[4],  w, h, mirror=True)
        index_tip   = lm_xy(lm[8],  w, h, mirror=True)
        index_mcp   = lm_xy(lm[5],  w, h, mirror=True)
        middle_tip  = lm_xy(lm[12], w, h, mirror=True)
        middle_mcp  = lm_xy(lm[9],  w, h, mirror=True)

        index_tip_px = index_tip

    
        idx_len = euclidean(index_mcp, index_tip) + 1e-6
        pinch_ratio_idx = euclidean(thumb_tip, index_tip) / idx_len
        left_pinch_now = pinch_ratio_idx < PINCH_RATIO_THRESHOLD

        mid_len = euclidean(middle_mcp, middle_tip) + 1e-6
        pinch_ratio_mid = euclidean(thumb_tip, middle_tip) / mid_len
        right_pinch_now = pinch_ratio_mid < RIGHT_PINCH_RATIO

     
        avg_len = 0.5 * (idx_len + mid_len)
        two_finger_ratio = euclidean(index_tip, middle_tip) / avg_len
        two_fingers_now = (two_finger_ratio < TWO_FINGER_RATIO_THRESHOLD) and (not left_pinch_now) and (not right_pinch_now)

   
        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )
        cv2.putText(frame, f"idx_pinch: {pinch_ratio_idx:.2f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        cv2.putText(frame, f"mid_pinch: {pinch_ratio_mid:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        cv2.putText(frame, f"2f_ratio: {two_finger_ratio:.2f}", (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

 
    if index_tip_px is not None:
        smoothed_cursor = smooth_point(smoothed_cursor, index_tip_px, alpha=0.25)
        cx, cy = smoothed_cursor
        cursor_trail.append((int(cx), int(cy)))
        sx, sy = map_to_screen(cx, cy, w, h)

        if not scroll_mode:
            pyautogui.moveTo(sx, sy)


    if two_fingers_now:
        if not scroll_mode:
            scroll_mode = True
            prev_scroll_y = smoothed_cursor[1] if smoothed_cursor is not None else None
            scroll_gap_frames = 0
        else:
            scroll_gap_frames = 0
            if smoothed_cursor is not None and prev_scroll_y is not None:
                dy = smoothed_cursor[1] - prev_scroll_y
                amt = int(-dy * SCROLL_GAIN)
                if amt != 0:
                    pyautogui.scroll(amt)
                prev_scroll_y = smoothed_cursor[1]
    else:
        if scroll_mode:
            scroll_gap_frames += 1
            if scroll_gap_frames >= SCROLL_DEBOUNCE_FRAMES:
                scroll_mode = False
                prev_scroll_y = None
                scroll_gap_frames = 0

    now_ms = int(time.time() * 1000)

    if not scroll_mode:
        if left_pinch_now and not left_pinch_active:
            left_pinch_active = True
            left_pinch_start_ms = now_ms
            pyautogui.mouseDown()
            is_left_dragging = True

        if (not left_pinch_now) and left_pinch_active:
            left_pinch_active = False
            dur = now_ms - left_pinch_start_ms

            if is_left_dragging:
                pyautogui.mouseUp()
                is_left_dragging = False

            if dur < QUICK_PINCH_MS:
                pyautogui.click()


    if right_pinch_now and not right_pinch_active and right_click_armed and not scroll_mode:
        right_pinch_active = True
        right_click_armed = False
        pyautogui.click(button='right')

    if (not right_pinch_now) and right_pinch_active:
        right_pinch_active = False
        right_click_armed = True

    status = []
    if scroll_mode:
        status.append("SCROLL")
    else:
        if is_left_dragging:
            status.append("DRAG")
        elif left_pinch_now:
            status.append("PINCH")
        else:
            status.append("OPEN")
        if right_pinch_now:
            status.append("RIGHT")

    cv2.putText(frame, f"Status: {'+'.join(status)}", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if len(cursor_trail) > 1:
        for i in range(1, len(cursor_trail)):
            cv2.line(frame, cursor_trail[i-1], cursor_trail[i], (200, 200, 200), 2)
        cv2.circle(frame, cursor_trail[-1], 8, (255, 255, 255), -1)

    cv2.imshow("Hand Mouse + Scroll (Windows) — press 'x' to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
