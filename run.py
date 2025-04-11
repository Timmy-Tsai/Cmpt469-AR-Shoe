import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import trimesh

# Define encoderLight, decoderLight, and LightNet classes (unchanged)
class encoderLight(nn.Module):
    def __init__(self, SGNum=12, cascadeLevel=0):
        super(encoderLight, self).__init__()
        self.SGNum = SGNum
        self.cascadeLevel = cascadeLevel
        self.preProcess = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(11, 32, kernel_size=4, stride=2, bias=True),
            nn.GroupNorm(2, 32),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=True),
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True)
        )
        self.pad1 = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(64 + (SGNum * 7 if cascadeLevel != 0 else 0), 128, kernel_size=4, stride=2, bias=True)
        self.gn1 = nn.GroupNorm(8, 128)
        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, bias=True)
        self.gn2 = nn.GroupNorm(16, 256)
        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=4, stride=2, bias=True)
        self.gn3 = nn.GroupNorm(16, 256)
        self.pad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, bias=True)
        self.gn4 = nn.GroupNorm(32, 512)
        self.pad5 = nn.ZeroPad2d(1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, bias=True)
        self.gn5 = nn.GroupNorm(32, 512)
        self.pad6 = nn.ReplicationPad2d(1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, bias=True)
        self.gn6 = nn.GroupNorm(64, 1024)

    def forward(self, x):
        x = self.preProcess(x)
        x1 = F.relu(self.gn1(self.conv1(self.pad1(x))), True)
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1))), True)
        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2))), True)
        x4 = F.relu(self.gn4(self.conv4(self.pad4(x3))), True)
        x5 = F.relu(self.gn5(self.conv5(self.pad5(x4))), True)
        x6 = F.relu(self.gn6(self.conv6(self.pad6(x5))), True)
        return x1, x2, x3, x4, x5, x6

class decoderLight(nn.Module):
    def __init__(self, SGNum=12, outChannel=36):
        super(decoderLight, self).__init__()
        self.SGNum = SGNum
        self.outChannel = outChannel
        self.dconv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=True)
        self.dgn1 = nn.GroupNorm(32, 512)
        self.dconv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=True)
        self.dgn2 = nn.GroupNorm(32, 512)
        self.dconv3 = nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(16, 256)
        self.dconv4 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True)
        self.dgn4 = nn.GroupNorm(16, 256)
        self.dconv5 = nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=True)
        self.dgn5 = nn.GroupNorm(8, 128)
        self.dconv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True)
        self.dgn6 = nn.GroupNorm(8, 128)
        self.dconvFinal = nn.Conv2d(128, outChannel, kernel_size=3, padding=1, bias=True)

    def forward(self, x1, x2, x3, x4, x5, x6):
        x1 = F.interpolate(x1, size=(1, 2), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(1, 2), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(1, 2), mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=(1, 2), mode='bilinear', align_corners=False)
        x5 = F.interpolate(x5, size=(1, 2), mode='bilinear', align_corners=False)
        xd1 = F.relu(self.dgn1(self.dconv1(x6)))
        xd1 = torch.cat([xd1, x5], dim=1)
        xd2 = F.relu(self.dgn2(self.dconv2(xd1)))
        xd2 = torch.cat([xd2, x4], dim=1)
        xd3 = F.relu(self.dgn3(self.dconv3(xd2)))
        xd3 = torch.cat([xd3, x3], dim=1)
        xd4 = F.relu(self.dgn4(self.dconv4(xd3)))
        xd4 = torch.cat([xd4, x2], dim=1)
        xd5 = F.relu(self.dgn5(self.dconv5(xd4)))
        xd5 = torch.cat([xd5, x1], dim=1)
        xd6 = F.relu(self.dgn6(self.dconv6(xd5)))
        out = torch.tanh(self.dconvFinal(xd6))
        return out

encoderLight.__module__ = 'models'
decoderLight.__module__ = 'models'

class LightNet(nn.Module):
    def __init__(self, SGNum=12):
        super(LightNet, self).__init__()
        self.SGNum = SGNum
        self.encoder = encoderLight(SGNum=SGNum, cascadeLevel=0)
        self.axis_decoder = decoderLight(SGNum=SGNum, outChannel=3*SGNum)
        self.lamb_decoder = decoderLight(SGNum=SGNum, outChannel=SGNum)
        self.weight_decoder = decoderLight(SGNum=SGNum, outChannel=3*SGNum)

        def load_component(component, path):
            try:
                full_model = torch.load(path, map_location='cpu', weights_only=False)
                component.load_state_dict(full_model.state_dict())
                print(f"Successfully loaded weights from {path}")
            except Exception as e:
                raise RuntimeError(f"Error loading weights from {path}: {str(e)}")
        
        load_component(self.encoder, 'models/check_cascadeLight0_sg12_offset1.0/lightEncoder0_9.pth')
        load_component(self.axis_decoder, 'models/check_cascadeLight0_sg12_offset1.0/axisDecoder0_9.pth')
        load_component(self.lamb_decoder, 'models/check_cascadeLight0_sg12_offset1.0/lambDecoder0_9.pth')
        load_component(self.weight_decoder, 'models/check_cascadeLight0_sg12_offset1.0/weightDecoder0_9.pth')

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = self.encoder(x)
        xi = self.axis_decoder(x1, x2, x3, x4, x5, x6)
        lamb = self.lamb_decoder(x1, x2, x3, x4, x5, x6)
        weight = self.weight_decoder(x1, x2, x3, x4, x5, x6)
        return torch.cat([xi, lamb, weight], dim=1)

def process_image(image_path, shoe_model_path, texture_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    height, width = image.shape[:2]

    # Pose detection
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            raise ValueError("No person detected in the image")
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        left_toe = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        left_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]

    ankle_pixel = (int(left_ankle.x * width), int(left_ankle.y * height))
    toe_pixel = (int(left_toe.x * width), int(left_toe.y * height))
    heel_pixel = (int(left_heel.x * width), int(left_heel.y * height))

    # Calculate foot dimensions in pixels
    foot_length_pixels = np.linalg.norm(np.array(toe_pixel) - np.array(heel_pixel))
    ankle_to_heel_pixels = np.linalg.norm(np.array(ankle_pixel) - np.array(heel_pixel))
    ankle_to_toe_pixels = np.linalg.norm(np.array(ankle_pixel) - np.array(toe_pixel))
    print(f"Foot length: {foot_length_pixels:.2f}px, Ankle to heel: {ankle_to_heel_pixels:.2f}px, Ankle to toe: {ankle_to_toe_pixels:.2f}px")

    # Lighting prediction (unchanged)
    def preprocess_image(img):
        img = cv2.resize(img, (320, 240))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    input_tensor = preprocess_image(image)
    model = LightNet(SGNum=12)
    model.eval()
    with torch.no_grad():
        light_input = torch.cat([input_tensor, torch.zeros(1, 8, 240, 320)], dim=1)
        lighting_pred = model(light_input)

    scale_factor_y = 1 / height
    scale_factor_x = 2 / width
    i = int(ankle_pixel[1] * scale_factor_y)
    j = int(ankle_pixel[0] * scale_factor_x)
    i = np.clip(i, 0, 0)
    j = np.clip(j, 0, 1)
    lighting_params = lighting_pred[0, :, i, j].numpy()
    xi_raw = lighting_params[:36].reshape(12, 3)
    lambda_raw = lighting_params[36:48]
    F_raw = lighting_params[48:].reshape(12, 3)
    xi = xi_raw / np.linalg.norm(xi_raw, axis=1, keepdims=True)
    lambda_k = np.tan(np.pi / 4 * (lambda_raw + 1))
    F_k = np.tan(np.pi / 4 * (F_raw + 1))

    env_height, env_width = 512, 1024
    theta = np.linspace(0, np.pi, env_height)
    phi = np.linspace(-np.pi, np.pi, env_width)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    directions = np.stack([np.sin(theta_grid) * np.cos(phi_grid), np.sin(theta_grid) * np.sin(phi_grid), np.cos(theta_grid)], axis=-1)
    env_map = np.zeros((env_height, env_width, 3), dtype=np.float32)
    for k in range(12):
        dot_product = np.sum(directions * xi[k], axis=-1)
        lobe = np.exp(lambda_k[k] * (dot_product - 1))
        env_map += F_k[k] * lobe[..., np.newaxis]
    env_map = env_map / env_map.max()

    # Load shoe mesh with Open3D
    mesh = o3d.io.read_triangle_mesh(shoe_model_path)
    if not mesh.has_triangles():
        raise ValueError("No triangles in mesh")
    mesh.compute_vertex_normals()
    if not mesh.has_triangle_uvs():
        raise ValueError("Mesh has no UV coordinates")
    
    uvs = np.asarray(mesh.triangle_uvs)
    uvs[:, 1] = 1.0 - uvs[:, 1]  # Flip vertically
    texture_img = o3d.io.read_image(texture_path)
    if texture_img.is_empty():
        raise ValueError("Failed to load texture")
    mesh.textures = [texture_img]
    mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(mesh.triangles))

    # Center and scale mesh
    vertices = np.asarray(mesh.vertices)
    centroid = vertices.mean(axis=0)
    vertices -= centroid

    # Calculate shoe dimensions in mesh units
    shoe_length = vertices[:, 0].max() - vertices[:, 0].min()
    shoe_width = vertices[:, 2].max() - vertices[:, 2].min()
    shoe_height = vertices[:, 1].max() - vertices[:, 1].min()
    print(f"Original shoe dimensions in mesh units: length={shoe_length:.2f}, width={shoe_width:.2f}, height={shoe_height:.2f}")

    # Initial scale to 2-unit box
    bounds_range = np.max(vertices.max(axis=0) - vertices.min(axis=0))
    initial_scale = 2.0 / bounds_range if bounds_range > 0 else 1.0
    vertices *= initial_scale
    shoe_length *= initial_scale
    shoe_width *= initial_scale
    shoe_height *= initial_scale
    print(f"Shoe dimensions after initial scaling: length={shoe_length:.2f}, width={shoe_width:.2f}, height={shoe_height:.2f} units")

    # Estimate pixels per unit (assuming 2 units ≈ 512 pixels with zoom=0.8)
    pixels_per_unit = 512 / 2  # 256 pixels per unit
    desired_shoe_length_units = foot_length_pixels / pixels_per_unit
    additional_scale = desired_shoe_length_units / shoe_length
    # Add a size multiplier to increase the shoe's size
    size_multiplier = 2.0  # Increase size by 100% (double the size)
    additional_scale *= size_multiplier
    vertices *= additional_scale
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Verify the new dimensions after additional scaling
    new_vertices = np.asarray(mesh.vertices)
    new_shoe_length = new_vertices[:, 0].max() - new_vertices[:, 0].min()
    new_shoe_width = new_vertices[:, 2].max() - new_vertices[:, 2].min()
    new_shoe_height = new_vertices[:, 1].max() - new_vertices[:, 1].min()
    print(f"Additional scale factor (with multiplier {size_multiplier}): {additional_scale:.2f}")
    print(f"Final shoe dimensions: length={new_shoe_length:.2f}, width={new_shoe_width:.2f}, height={new_shoe_height:.2f} units")

    # Open3D offscreen rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, visible=False)
    vis.add_geometry(mesh)
    render_option = vis.get_render_option()
    render_option.light_on = True
    render_option.background_color = np.array([0, 0, 0])

    # Camera setup - adjust zoom to accommodate larger shoe
    ctr = vis.get_view_control()
    # Decrease zoom to fit the larger shoe in the frame (smaller value = zoom out)
    ctr.set_zoom(0.5)
    ctr.set_front([0, 1, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])

    vis.poll_events()
    vis.update_renderer()
    color = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    depth = np.asarray(vis.capture_depth_float_buffer(do_render=True))
    vis.destroy_window()

    # Convert to uint8
    color = (color * 255).astype(np.uint8)
    depth = (depth > 0).astype(np.uint8)
    shoe_image = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    # Save raw render
    cv2.imwrite("raw_shoe_render.jpg", shoe_image)

    # Debug: Check the size of the shoe in the rendered image
    non_zero_pixels = np.sum(depth > 0)
    print(f"Number of non-zero pixels in depth buffer (indicating shoe size in render): {non_zero_pixels}")

    # Affine transform
    heel_toe_vec = np.array([520 - 120, 0])  # [400, 0]
    perp_vec = np.array([-heel_toe_vec[1], heel_toe_vec[0]])  # [0, 400]
    src_points = np.float32([
        [120, 240],              # heel
        [520, 240],              # toe
        [120 + perp_vec[0], 240 + perp_vec[1]]  # perpendicular
    ])
    heel_toe_img = np.array(toe_pixel) - np.array(heel_pixel)
    perp_img = np.array([-heel_toe_img[1], heel_toe_img[0]])
    # Define destination points and apply a y-offset to move the shoe up
    y_offset = -15  # Move up by 15 pixels (negative y direction)
    dst_points = np.float32([
        [heel_pixel[0], heel_pixel[1] + y_offset],  # heel
        [toe_pixel[0], toe_pixel[1] + y_offset],    # toe
        [heel_pixel[0] + perp_img[0], heel_pixel[1] + perp_img[1] + y_offset]  # perpendicular
    ])

    M = cv2.getAffineTransform(src_points, dst_points)
    warped_shoe = cv2.warpAffine(shoe_image, M, (width, height), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(depth, M, (width, height), flags=cv2.INTER_NEAREST) > 0

    # Debug: Check the size of the shoe in the warped image
    warped_non_zero_pixels = np.sum(warped_mask)
    print(f"Number of non-zero pixels in warped mask (indicating shoe size in final image): {warped_non_zero_pixels}")

    # Composite
    composite = image.copy()
    composite[warped_mask] = warped_shoe[warped_mask]
    return composite

if __name__ == "__main__":
    input_image = "fullbodynoshoe1.jpg"
    shoe_model = "NikeShoe_01_MS_Cleaned_Positioned_Final_2.obj"
    texture_path = "NikeShoe_01_MS_Cleaned_Positioned_Final_2.jpg"
    
    try:
        result = process_image(input_image, shoe_model, texture_path)
        cv2.imwrite("output.jpg", result)
        print("Successfully generated output.jpg")
    except Exception as e:
        print(f"Error processing image: {str(e)}")