import mediapipe as mp
import cv2
import numpy as np
import threading
import time 

cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

##this is thread for count
def count_up():
    pre_positions=combine()
    positions=pre_positions
    status="UP"
    time.sleep(1)
    def pos_diff():
        return pre_positions[1]-positions[1]
    while not stop_thread:
        if pos_diff()<= -0.006 and status=="UP" and pos_diff()<=0:
            status="DOWN"
            global count
            count =count+1
        elif pos_diff()>=0.006 and status=="DOWN" and pos_diff()>=0:
            status="UP"
        time.sleep(0.01)
        pre_positions=positions
        positions=combine()

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 
def combine():
     return [left_wrist[1],right_wrist[1]]
##screen_width, screen_height = get_screen_size()
stop_thread=False
left_wrist=[0,0]
right_wrist=[0,0]

count = 0
bg_thread = threading.Thread(target=count_up)
bg_thread.start()
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image=cv2.resize(image,(0,0),fx=1.5,fy=1.5)
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            # Calculate angle
            right_elbow_angle=calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_elbow_angle=calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_knee_angle=calculate_angle(right_hip, right_knee, right_ankle)
            left_knee_angle=calculate_angle(left_hip, left_knee, left_ankle)
            
            # Visualize angle
            cv2.putText(image, str(right_elbow_angle), 
                           tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(right_knee_angle), 
                           tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                                )   
            cv2.putText(image, str(left_elbow_angle), 
                           tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(left_knee_angle), 
                           tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                                )
            if  left_elbow_angle<=150:
                cv2.putText(image,"don't bend your left elbow",
                            tuple(np.multiply((0,0.5), [640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255), 2, cv2.LINE_AA
                            )
            if  left_knee_angle<=40:
                cv2.putText(image,"Move forward",
                            tuple(np.multiply((0,0.7), [640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255), 2, cv2.LINE_AA
                            )   
            if left_knee_angle>=70:
                            cv2.putText(image,"Move backward",
                            tuple(np.multiply((0,0.7), [640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255), 2, cv2.LINE_AA
                            )   
            if  right_elbow_angle<=150:
                cv2.putText(image,"don't bend your right elbow",
                            tuple(np.multiply((0.7,0.5), [640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255), 2, cv2.LINE_AA
                            )
            if  right_knee_angle<=40:
                cv2.putText(image,"Move forward",
                            tuple(np.multiply((0.7,0.7), [640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255), 2, cv2.LINE_AA
                            )   
            if right_knee_angle>70:
                            cv2.putText(image,"Move backward",
                            tuple(np.multiply((0.7,0.7), [640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255), 2, cv2.LINE_AA
                            )
            cv2.putText(image,"count "+str(count),
                            tuple(np.multiply((0.8,0.2), [640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255), 2, cv2.LINE_AA
                            )   
            
            
            
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)
        key=cv2.waitKey(10) & 0xFF
        if  key == ord('q'):
            stop_thread=True
            break
        elif key == ord('s'):
            count=0


    cap.release()
    cv2.destroyAllWindows()