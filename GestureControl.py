import cv2
import mediapipe as mp
import pyautogui as auto

cam_w, cam_h = 600, 480
scr_w, scr_h = auto.size()
start_pt = 50

cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 1, 1, 0.7, 0.5)
mpDraw = mp.solutions.drawing_utils

def coordinates(lm, dim, scr):
    return (int(((lm * (dim + start_pt) ) - start_pt) / (dim - start_pt) * (scr)))

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(RGB_frame)

    h, w, c = frame.shape                
    h -= start_pt
    w -= start_pt
    index7_y = 0

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:            
            for id, lm in enumerate(handLms.landmark):                                
                if id ==7:
                    index7_y = coordinates(lm.y, h, scr_h)
                elif id == 4:
                    thumb_y = coordinates(lm.y, h, scr_h)
                                       
                elif id == 8:
                    index_x = coordinates(lm.x, w, scr_w)
                    index_y = coordinates(lm.y, h, scr_h)
                    cv2.circle(frame, (int(lm.x*(w+start_pt)), int(lm.y*(h+start_pt))), 18, (255, 0, 200), -1)
                    auto.moveTo(index_x, index_y)
                elif id == 12:
                    mid_y = coordinates(lm.y, h, scr_h)
                    if (mid_y < index7_y):
                        auto.click(button='right')
                        auto.sleep(0.5)
                elif id == 16:
                    ring_y = coordinates(lm.y, h, scr_h)
                    if ((ring_y < index7_y)):
                        auto.scroll(10)
                elif id == 20:
                    little_y = coordinates(lm.y, h, scr_h)
                    if (little_y < index7_y):
                        auto.scroll(-10)

            if ((thumb_y - index7_y) < 50):
                auto.click()
                auto.sleep(0.2) 

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)    

    frame = cv2.rectangle(frame, (start_pt, start_pt), (w, h), (255, 0, 200), 3)    
    cv2.imshow('System Control using Hands', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()