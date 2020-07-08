import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time
from models.resfcn import load_SPRNET
from data.dataset import USRTestDataset, ToTensor
from torch.utils.data import DataLoader
from math import cos, sin, atan2, asin
from skimage import io
import matplotlib.pylab as plt
from data.processing.faceutil import mesh
working_folder  = "/home/viet/Projects/Pycharm/SPRNet/"
FLAGS = {
            "model"             : os.path.join(working_folder, "train_log/_checkpoint_epoch_80.pth.tar"),
            "data_path"         : os.path.join(working_folder, "data"),
			"uv_kpt_ind_path"   : os.path.join(working_folder, "data/processing/Data/UV/uv_kpt_ind.txt"),
            "face_ind_path"     : os.path.join(working_folder, "data/processing/Data/UV/face_ind.txt"),
            "triangles_path"    : os.path.join(working_folder, "data/processing/Data/UV/triangles.txt"),
            "canonical_vts_path": os.path.join(working_folder, "data/processing/Data/UV/canonical_vertices.npy"),
			"result_path"       : os.path.join(working_folder, "result/usr"),
            "uv_kpt_path"       : os.path.join(working_folder, "data/processing/Data/UV/uv_kpt_ind.txt"),
            "uv_face_mask"      : os.path.join(working_folder, "data/processing/Data/UV/uv_face_mask.png"),
            "device"            : "cuda",
            "devices_id"        : [0],
            "batch_size"        : 16, 
            "workers"           : 8
		}

uv_kpt_ind          = np.loadtxt(FLAGS["uv_kpt_ind_path"]).astype(np.int32)
face_ind            = np.loadtxt(FLAGS["face_ind_path"]).astype(np.int32)
triangles           = np.loadtxt(FLAGS["triangles_path"]).astype(np.int32)
canonical_vertices  = np.load(FLAGS["canonical_vts_path"])
face_mask_np        = io.imread(FLAGS["uv_face_mask"]) / 255.
resolution  = 256

def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''
    # assert(isRotationMatrix(R))

    if R[2, 0] != 1 or R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    return x, y, z

def P2sRt(P):
    ''' decomposing camera matrix P. 
    Args: 
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation. 
    '''
    t2d = P[:2, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t2d

def compute_similarity_transform(points_static, points_to_transform):
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T

    t0 = -np.mean(p0, axis=1).reshape(3, 1)
    t1 = -np.mean(p1, axis=1).reshape(3, 1)
    t_final = t1 - t0

    p0c = p0 + t0
    p1c = p1 + t1

    covariance_matrix = p0c.dot(p1c.T) #3 3
    U, S, V = np.linalg.svd(covariance_matrix) #U 3 3 S 3 V 3 3
    R = U.dot(V) #R 3 3
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0) ** 2))
    rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0) ** 2))

    s = (rms_d0 / rms_d1)
    P = np.c_[s * np.eye(3).dot(R), t_final]
    temp = np.eye(3).dot(R)
    P_= np.c_[s * temp, t_final]
    return P

def estimate_pose(vertices):
    P = compute_similarity_transform(vertices, canonical_vertices)
    s, R, t = P2sRt(P)
    pose = matrix2angle(R)

    return P, pose, (s, R, t)

def frontalize(vertices):
    vertices_homo = np.hstack((vertices, np.ones([vertices.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(vertices_homo, canonical_vertices)[0].T # Affine matrix. 3 x 4
    front_vertices = vertices_homo.dot(P.T)

    return front_vertices

def get_landmarks(pos):
    kpt = pos[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
    return kpt

def get_vertices(pos):
    all_vertices = np.reshape(pos, [resolution**2, -1])
    vertices = all_vertices[face_ind, :]
    return vertices

def get_colors_from_texture(texture):
    all_colors = np.reshape(texture, [resolution**2, -1])
    colors = all_colors[face_ind, :]
    return colors

def get_colors(image, vertices):
    [h, w, _] = image.shape
    vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)
    vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)
    ind = np.round(vertices).astype(np.int32)
    colors = image[ind[:,1], ind[:,0], :]
    return colors

def show_uv_mesh(img, uv, isMesh=True):
    img = cv2.resize(img, (256,256))
    img = cv2.resize(img, None, fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
    if isMesh:
        x, y, z = get_vertices(uv).transpose() * 2
        for i in range(0, x.shape[0], 1):
            img = cv2.circle(img, (int(x[i]), int(y[i])), 1, (255, 0, 0), -1)
    keypoint = uv[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
    x, y, z = keypoint.transpose().astype(np.int32) * 2
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]), int(y[i])), 4, (255, 255, 255), -1)
    cv2.imshow("uv_point_scatter",img)
    cv2.waitKey()

def show_kpt_result(img, prd, grt):
    # img = cv2.resize(img, (256,256))
    img = cv2.resize(img, None, fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
    uv_kpt             = prd[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
    gt_kpt             = grt[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
    x_uv, y_uv, z_uv   = uv_kpt.transpose().astype(np.int32) * 2
    x_gt, y_gt, z_gt   = gt_kpt.transpose().astype(np.int32) * 2
    for i in range(68):
        img = cv2.circle(img, (int(x_uv[i]), int(y_uv[i])), 4, (255, 255, 255), -1)
        img = cv2.circle(img, (int(x_gt[i]), int(y_gt[i])), 4, (0, 255, 0), -1)
    cv2.imshow("uv_point_scatter",img)
    cv2.waitKey()

def UVmap2Mesh(uv_position_map, uv_texture_map=None, only_foreface=True, is_extra_triangle=False):
    """
    if no texture map is provided, translate the position map to a point cloud
    :param uv_position_map:
    :param uv_texture_map:
    :param only_foreface:
    :return: mesh data
    """
    [uv_h, uv_w, uv_c] = [256, 256, 3]
    vertices = []
    colors = []
    triangles = []
    if uv_texture_map is not None:
        for i in range(uv_h):
            for j in range(uv_w):
                if not only_foreface:
                    vertices.append(uv_position_map[i][j])
                    colors.append(uv_texture_map[i][j])
                    pa = i * uv_h + j
                    pb = i * uv_h + j + 1
                    pc = (i - 1) * uv_h + j
                    pd = (i + 1) * uv_h + j + 1
                    if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                        triangles.append([pa, pb, pc])
                        triangles.append([pa, pc, pb])
                        triangles.append([pa, pb, pd])
                        triangles.append([pa, pd, pb])

                else:
                    if face_mask_np[i, j] == 0:
                        vertices.append(np.array([0, 0, 0]))
                        colors.append(np.array([0, 0, 0]))
                        continue
                    else:
                        vertices.append(uv_position_map[i][j])
                        colors.append(uv_texture_map[i][j])
                        pa = i * uv_h + j
                        pb = i * uv_h + j + 1
                        pc = (i - 1) * uv_h + j
                        pd = (i + 1) * uv_h + j + 1
                        if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                            if is_extra_triangle:
                                pe = (i - 1) * uv_h + j + 1
                                pf = (i + 1) * uv_h + j
                                if (face_mask_np[i, j + 1] > 0) and (face_mask_np[i + 1, j + 1] > 0) and (face_mask_np[i + 1, j] > 0) and (
                                        face_mask_np[i - 1, j + 1] > 0 and face_mask_np[i - 1, j] > 0):
                                    triangles.append([pa, pb, pc])
                                    triangles.append([pa, pc, pb])
                                    triangles.append([pa, pc, pe])
                                    triangles.append([pa, pe, pc])
                                    triangles.append([pa, pb, pe])
                                    triangles.append([pa, pe, pb])
                                    triangles.append([pb, pc, pe])
                                    triangles.append([pb, pe, pc])

                                    triangles.append([pa, pb, pd])
                                    triangles.append([pa, pd, pb])
                                    triangles.append([pa, pb, pf])
                                    triangles.append([pa, pf, pb])
                                    triangles.append([pa, pd, pf])
                                    triangles.append([pa, pf, pd])
                                    triangles.append([pb, pd, pf])
                                    triangles.append([pb, pf, pd])

                            else:
                                if not face_mask_np[i, j + 1] == 0:
                                    if not face_mask_np[i - 1, j] == 0:
                                        triangles.append([pa, pb, pc])
                                        triangles.append([pa, pc, pb])
                                    if not face_mask_np[i + 1, j + 1] == 0:
                                        triangles.append([pa, pb, pd])
                                        triangles.append([pa, pd, pb])
    else:
        for i in range(uv_h):
            for j in range(uv_w):
                if not only_foreface:
                    vertices.append(uv_position_map[i][j])
                    colors.append(np.array([64, 64, 64]))
                    pa = i * uv_h + j
                    pb = i * uv_h + j + 1
                    pc = (i - 1) * uv_h + j
                    if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                        triangles.append([pa, pb, pc])
                else:
                    if face_mask_np[i, j] == 0:
                        vertices.append(np.array([0, 0, 0]))
                        colors.append(np.array([0, 0, 0]))
                        continue
                    else:
                        vertices.append(uv_position_map[i][j])
                        colors.append(np.array([128, 0, 128]))
                        pa = i * uv_h + j
                        pb = i * uv_h + j + 1
                        pc = (i - 1) * uv_h + j
                        if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                            if not face_mask_np[i, j + 1] == 0:
                                if not face_mask_np[i - 1, j] == 0:
                                    triangles.append([pa, pb, pc])
                                    triangles.append([pa, pc, pb])

    vertices = np.array(vertices)
    colors = np.array(colors)
    triangles = np.array(triangles)
    # verify_face = mesh.render.render_colors(verify_vertices, verify_triangles, verify_colors, height, width,
    #                                         channel)
    mesh_info = {   
                    'vertices': vertices, 
                    'triangles': triangles,
                    'full_triangles': triangles,
                    'colors': colors
                    }
    return mesh_info

def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: [u, v] or [x, y] 
        tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points.T

    # vectors
    v0 = tp[:,2] - tp[:,0]
    v1 = tp[:,1] - tp[:,0]
    v2 = point - tp[:,0]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)

def render_colors(vertices, triangles, colors, h, w, c = 3):
    ''' render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width    
    Returns:
        image: [h, w, c]. 
    '''
    assert vertices.shape[0] == colors.shape[0]
    
    # initial 
    image = np.zeros((h, w, c))
    depth_buffer = np.zeros([h, w]) - 999999.

    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vertices[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vertices[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2])
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]

                if point_depth > depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth
                    image[v, u, :] = w0*colors[tri[0], :] + w1*colors[tri[1], :] + w2*colors[tri[2], :]
    return image

def showMesh(mesh_info, tex, init_img=None):
    height = np.ceil(np.max(mesh_info['vertices'][:, 1])).astype(int)
    width = np.ceil(np.max(mesh_info['vertices'][:, 0])).astype(int)
    channel = 3
    if init_img is not None:
        [height, width, channel] = init_img.shape
    mesh_image = (mesh.render.render_colors(mesh_info['vertices'], mesh_info['full_triangles'], mesh_info['colors'],
                                            height, width, channel)).astype(np.uint8)
    
    if init_img is None:
        mesh_image = cv2.resize(mesh_image, (256,256))
        mesh_image = cv2.cvtColor(mesh_image, cv2.COLOR_BGR2RGB)
        io.imshow(mesh_image)
        plt.show()
    else:
        plt.subplot(1, 3, 1)
        # plt.title('Origin')
        init_img = cv2.resize(init_img, (256,256))
        init_img = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(init_img)

        plt.subplot(1, 3, 2)
        # plt.title('Texture')
        tex_img = cv2.resize(tex, (256,256))
        tex_img = cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(tex_img)

        plt.subplot(1, 3, 3)
        # plt.title('3D Render')
        mesh_image = cv2.resize(mesh_image, (256,256))
        mesh_image = cv2.cvtColor(mesh_image, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(mesh_image)
        
        plt.show()

def get_point_weight(point, tri_points):
    ''' Get the weights of the position
    Methods: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
     -m1.compute the area of the triangles formed by embedding the point P inside the triangle
     -m2.Christer Ericson's book "Real-Time Collision Detection". faster.(used)
    Args:
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices(2d points) of a triangle. 
    Returns:
        w0: weight of v0
        w1: weight of v1
        w2: weight of v3
     '''
    tp = tri_points
    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2

def render_texture(vertices, triangles, texture, tex_coords, tex_triangles, h, w, c = 3, mapping_type = 'bilinear'):
    ''' render mesh with texture map
    Args:
        vertices: [nver], 3
        triangles: [ntri, 3]
        texture: [tex_h, tex_w, 3]
        tex_coords: [ntexcoords, 3]
        tex_triangles: [ntri, 3]
        h: height of rendering
        w: width of rendering
        c: channel
        mapping_type: 'bilinear' or 'nearest'
    '''
    assert triangles.shape[0] == tex_triangles.shape[0]
    tex_h, tex_w, _ = texture.shape

    # initial 
    image = np.zeros((h, w, c))
    depth_buffer = np.zeros([h, w]) - 999999.

    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices
        tex_tri = tex_triangles[i, :] # 3 tex indice

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vertices[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vertices[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2])
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]
                if point_depth > depth_buffer[v, u]:
                    # update depth
                    depth_buffer[v, u] = point_depth    
                    
                    # tex coord
                    tex_xy = w0*tex_coords[tex_tri[0], :] + w1*tex_coords[tex_tri[1], :] + w2*tex_coords[tex_tri[2], :]
                    tex_xy[0] = max(min(tex_xy[0], float(tex_w - 1)), 0.0); 
                    tex_xy[1] = max(min(tex_xy[1], float(tex_h - 1)), 0.0); 

                    # nearest
                    if mapping_type == 'nearest':
                        tex_xy = np.round(tex_xy).astype(np.int32)
                        tex_value = texture[tex_xy[1], tex_xy[0], :] 

                    # bilinear
                    elif mapping_type == 'bilinear':
                        # next 4 pixels
                        ul = texture[int(np.floor(tex_xy[1])), int(np.floor(tex_xy[0])), :]
                        ur = texture[int(np.floor(tex_xy[1])), int(np.ceil(tex_xy[0])), :]
                        dl = texture[int(np.ceil(tex_xy[1])), int(np.floor(tex_xy[0])), :]
                        dr = texture[int(np.ceil(tex_xy[1])), int(np.ceil(tex_xy[0])), :]

                        yd = tex_xy[1] - np.floor(tex_xy[1])
                        xd = tex_xy[0] - np.floor(tex_xy[0])
                        tex_value = ul*(1-xd)*(1-yd) + ur*xd*(1-yd) + dl*(1-xd)*yd + dr*xd*yd

                    image[v, u, :] = tex_value
    return image

# def predict():
#     model               = load_SPRNET()
#     torch.cuda.set_device(FLAGS["devices_id"][0])
#     model	            = nn.DataParallel(model, device_ids=FLAGS["devices_id"]).cuda()
#     pretrained_weights  = torch.load(FLAGS["model"], map_location='cuda:0')
#     model.load_state_dict(pretrained_weights['state_dict'])
#     dataset 	        = USRTestDataset(
# 		root_dir  		= FLAGS["data_path"],
# 		transform 		= transforms.Compose([ToTensor()])
# 	)

#     data_loader = DataLoader(
# 		dataset, 
# 		batch_size      =   FLAGS["batch_size"], 
# 		num_workers     =   FLAGS["workers"],
# 		shuffle         =   False, 
# 		pin_memory      =   True, 
# 		drop_last       =   True
# 	)

#     cudnn.benchmark = True   
#     model.eval()
#     with torch.no_grad():
#         for i, (img) in tqdm(enumerate(data_loader)):
#             gens  = model(img, isTrain=False)[:, :, 1:257, 1:257]