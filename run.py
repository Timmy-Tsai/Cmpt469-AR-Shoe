import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.3
)

def detect_and_render_shoe(image_path, obj_path):
    # Load and process the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found")
        return
    
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        print("Detection failed. Try:")
        print("- Full-body/lower-body image")
        print("- Clear foot visibility")
        print("- Front/side view")
        return
    
    # Extract 2D foot landmarks
    FOOT_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.LEFT_HEEL,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX
    ]
    landmarks_2d = []
    for landmark in FOOT_LANDMARKS:
        idx = landmark.value
        lm = results.pose_landmarks.landmark[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        landmarks_2d.append([cx, cy])
    landmarks_2d = np.array(landmarks_2d, dtype=np.float32)
    
    # Load the shoe mesh with Open3D
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if not mesh.has_triangles():
        print("Error: No triangles found in the mesh.")
        return
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Define corresponding 3D points on the shoe model
    # Replace these with actual vertex indices from your mesh
    ankle_index = 100  # Example: vertex near ankle
    heel_index = 200   # Example: vertex near heel
    toe_index = 300    # Example: vertex near toe
    object_points = np.array([
        vertices[ankle_index],
        vertices[heel_index],
        vertices[toe_index]
    ], dtype=np.float32)
    
    # Define camera matrix (approximation)
    focal_length = w  # Adjust if needed
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # Assume no distortion
    
    # Estimate pose with solvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, landmarks_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
    )
    if not success:
        print("solvePnP failed")
        return
    
    # Project all 3D vertices to 2D
    projected_points, _ = cv2.projectPoints(
        vertices, rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2).astype(np.int32)
    
    # Create a blank canvas for rendering
    rendered_image = np.zeros_like(image)
    
    # Render each triangle
    for tri in triangles:
        pts = projected_points[tri].reshape(-1, 1, 2)
        cv2.fillPoly(rendered_image, [pts], color=(0, 255, 0))  # Green shoe
    
    # Overlay the rendered shoe on the original image
    alpha = 0.5  # Transparency (0.0 = fully transparent, 1.0 = fully opaque)
    cv2.addWeighted(rendered_image, alpha, image, 1 - alpha, 0, image)
    
    # Display the result
    cv2.imshow("Shoe on Foot", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "fullbodynoshoe.jpg"
    obj_path = "NikeShoe_01_MS_Cleaned_Positioned_Final_2.obj"
    detect_and_render_shoe(image_path, obj_path)