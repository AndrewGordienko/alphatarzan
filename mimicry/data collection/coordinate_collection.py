import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

f = open("coordinatedata.txt", "w")

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

cap = cv2.VideoCapture("sideagain.mp4")

_right_waist, _right_knee_angle, _right_ankle_angle = 0, 0, 0
_left_waist, _left_knee_angle, _left_ankle_angle = 0, 0, 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image = cv2.flip(image, 1)

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)
        
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        right_waist = calculate_angle(left_hip, right_hip, right_knee)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        left_waist = calculate_angle(right_hip, left_hip, left_knee)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        
        print("")

        if _right_waist != 0:
            if (_right_waist - right_waist) > 1 or (_right_knee_angle - right_knee_angle) > 1:
                print(f"previous right waist {_right_waist} prevoius right knee {_right_knee_angle}")
                print(f"right waist {right_waist} right knee {right_knee_angle}")
                print(f"previous left waist {_left_waist} previous left knee {_left_knee_angle}")
                print(f"left waist {left_waist} left knee {left_knee_angle}")

                f.write("-" + '\n')
                f.write(str((_right_waist, _right_knee_angle)) + '\n')
                f.write(str((right_waist, right_knee_angle)) + '\n')
                f.write(str((_left_waist, _left_knee_angle)) + '\n')
                f.write(str((left_waist, left_knee_angle)) + '\n')


        _right_waist, _right_knee_angle = right_waist, right_knee_angle
        _left_waist, _left_knee_angle = left_waist, left_knee_angle

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
