import open3d as o3d
import numpy as np

def show_shoe_model(obj_path, texture_path):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if not mesh.has_triangles():
        print("Error: No triangles found in the mesh.")
        return
    mesh.compute_vertex_normals()
    if not mesh.has_triangle_uvs():
        print("Warning: Mesh has no UV coordinates. Displaying without texture.")
        mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Red if no UVs
    else:
        # Debug UV coordinates
        uvs = np.asarray(mesh.triangle_uvs)
        uvs[:, 1] = 1.0 - uvs[:, 1]  # Flip vertically
        print("UV Coordinates Sample (first 5):", uvs[:5])
        print("UV Range:", uvs.min(), "to", uvs.max())
        
        # Load texture
        texture_img = o3d.io.read_image(texture_path)
        if texture_img.is_empty():
            print("Error: Texture image failed to load.")
            mesh.paint_uniform_color([0.0, 1.0, 0.0])  # Green if texture fails
        else:
            mesh.textures = [texture_img]
            mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(mesh.triangles))
    
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Textured Shoe Model",
        width=800,
        height=600,
        zoom=0.8
    )

if __name__ == "__main__":
    obj_path = "NikeShoe_01_MS_Cleaned_Positioned_Final_2.obj"
    texture_path = "NikeShoe_01_MS_Cleaned_Positioned_Final_2.jpg"  # Update to the correct texture path
    show_shoe_model(obj_path, texture_path)