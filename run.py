import cv2
import mediapipe as mp

# Initialize MediaPipe Pose with adjusted parameters
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,  # Use higher model complexity
    enable_segmentation=False,
    min_detection_confidence=0.3  # Lower confidence threshold
)

def detect_foot_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Focus on left foot landmarks
        FOOT_LANDMARKS = [
            mp_pose.PoseLandmark.LEFT_HEEL,
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            mp_pose.PoseLandmark.LEFT_ANKLE
        ]
        
        for landmark in FOOT_LANDMARKS:
            idx = landmark.value
            lm = results.pose_landmarks.landmark[idx]
            h, w = image.shape[:2]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 7, (0,255,0), -1)
        
        cv2.imshow("Result", image)
        cv2.waitKey(0)
    else:
        print("Detection failed. Try:")
        print("- Full-body/lower-body image")
        print("- Clear foot visibility")
        print("- Front/side view")

if __name__ == "__main__":
    detect_foot_landmarks("fullbodynoshoe.jpg")