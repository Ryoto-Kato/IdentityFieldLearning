import numpy as np
import torch

def tex2vertsColor(uv_map:np.ndarray, UV:np.ndarray, max_u, max_v):
    vertex_colors = []
    for i, tex_coord in enumerate(uv_map):
        # print(tex_coord)
        x, y = np.round(((1-tex_coord[1])*(max_u-1))%max_u).astype(np.int64), np.round(((tex_coord[0])*(max_v-1))%max_v).astype(np.int64)
        # print(x, y)
        color = UV[x, y, :]
        vertex_colors.append(color)
        # print(color)
    return np.asarray(vertex_colors)

def tex2vertsAttributes(uv_map, UV, max_u, max_v, index_in_batch):
    uv_map = uv_map[index_in_batch]
    num_verts = uv_map.shape[0]
    num_channels, _, _ = UV.shape

    # print(f"UV shape: {UV.shape}")
    # print(f"num_verts: {num_verts}")
    # print(f"num_channels: {num_channels}")

    vertex_attributes = torch.zeros(num_verts, num_channels).float()
    for i, tex_coord in enumerate(uv_map):
        # print(tex_coord)
        x, y = torch.round(((1-tex_coord[1])*(max_u-1))%max_u).long(), torch.round(((tex_coord[0])*(max_v-1))%max_v).long()
        # print(x, y)
        attrib = UV[:, x, y]
        vertex_attributes[i]=attrib
        # print(color)
    return vertex_attributes


def save_ply(f_name, vertices, faces, vertex_normals, vertex_colors, only_points = False):
    """
    Arguments
        vertices: np.array (N, 3)
        faces: np.array (F, 3)
        vertex_colors: np.array (N, 3)

    Returns
        ply file

    """
    
    header = "ply\n" + "format ascii 1.0\n"
    with open(f_name, 'w') as f:
        # write header
        f.write(header)
        # number of vertex
        num_verts = vertices.shape[0]
        f.write("element vertex "+str(num_verts)+"\n")
        # write vertex property
        verts_property = "property float x\n" + "property float y\n" + "property float z\n" + "property uchar red\n" + "property uchar green\n" + "property uchar blue\n"
        f.write(verts_property)

        # number of faces
        if not only_points:
            num_faces = faces.shape[0]
            f.write("element face "+str(num_faces) + "\n")
            # write face property
            faces_property = "property list uchar int vertex_index\n"
            f.write(faces_property)

        f.write("end_header\n")

        # write vertex location
        for coord, normal, color in zip(vertices, vertex_normals,vertex_colors):
            v_line = ""
            v_line = str(coord[0]) + " " + str(coord[1]) + " " + str(coord[2]) + " " + str(np.round(color[0]).astype(np.int64)) + " " + str(np.round(color[1]).astype(np.int64)) + " " + str(np.round(color[2]).astype(np.int64)) + "\n"
            f.write(v_line)

        if not only_points:
        # write triangle list
            for index3 in faces:
                t_line = ""
                t_line = "3 " + str(index3[0]) + " " + str(index3[1]) + " " + str(index3[2]) + "\n"
                f.write(t_line)
    f.close()