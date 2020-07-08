import math
import os
import numpy as np
import cv2
import skimage.transform
from scipy.io import loadmat
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import torchvision
from math import cos, sin, atan2, asin
import scipy.misc

def get_vertices(pos):
    all_vertices = np.reshape(pos, [resolution**2, -1])
    vertices = all_vertices[face_ind, :]
    return vertices

def get_landmarks(pos):
    kpt = pos[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
    return kpt

#region RENDER
def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices(2d points) of a triangle. 
    Returns:
        bool: true for in triangle
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

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)

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

def rasterize_triangles(vertices, triangles, h, w):
    ''' 
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        h: height
        w: width
    Returns:
        depth_buffer: [h, w] saves the depth, here, the bigger the z, the fronter the point.
        triangle_buffer: [h, w] saves the tri id(-1 for no triangle). 
        barycentric_weight: [h, w, 3] saves corresponding barycentric weight.

    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    # h, w is the size of rendering
    '''
    # initial 
    depth_buffer = np.zeros([h, w]) - 999999. #+ np.min(vertices[2,:]) - 999999. # set the initial z to the farest position
    triangle_buffer = np.zeros([h, w], dtype = np.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
    barycentric_weight = np.zeros([h, w, 3], dtype = np.float32)  # 
    
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
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2]) # barycentric weight
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]
                if point_depth > depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth
                    triangle_buffer[v, u] = i
                    barycentric_weight[v, u, :] = np.array([w0, w1, w2])

    return depth_buffer, triangle_buffer, barycentric_weight

def render_colors_ras(vertices, triangles, colors, h, w, c = 3):
    ''' render mesh with colors(rasterize triangle first)
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width    
        c: channel
    Returns:
        image: [h, w, c]. rendering.
    '''
    assert vertices.shape[0] == colors.shape[0]

    depth_buffer, triangle_buffer, barycentric_weight = rasterize_triangles(vertices, triangles, h, w)

    triangle_buffer_flat = np.reshape(triangle_buffer, [-1]) # [h*w]
    barycentric_weight_flat = np.reshape(barycentric_weight, [-1, c]) #[h*w, c]
    weight = barycentric_weight_flat[:, :, np.newaxis] # [h*w, 3(ver in tri), 1]

    colors_flat = colors[triangles[triangle_buffer_flat, :], :] # [h*w(tri id in pixel), 3(ver in tri), c(color in ver)]
    colors_flat = weight*colors_flat # [h*w, 3, 3]
    colors_flat = np.sum(colors_flat, 1) #[h*w, 3]. add tri.

    image = np.reshape(colors_flat, [h, w, c])
    # mask = (triangle_buffer[:,:] > -1).astype(np.float32)
    # image = image*mask[:,:,np.newaxis]
    return image

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

#endregion

#region POSE
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

def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: yaw.
        y: pitch.
        z: roll.
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x, y, z = angles[0], angles[1], angles[2]
    y, x, z = angles[0], angles[1], angles[2]

    # x
    Rx=np.array([   
                    [1,      0,       0],
                    [0, cos(x),  -sin(x)],
                    [0, sin(x),  cos(x)]
                ])
    # y
    Ry=np.array([   
                    [ cos(y), 0, sin(y)],
                    [      0, 1,      0],
                    [-sin(y), 0, cos(y)]
                ])
    # z
    Rz=np.array([   
                    [cos(z), -sin(z), 0],
                    [sin(z),  cos(z), 0],
                    [     0,       0, 1]
                ])
    R = Rz.dot(Ry).dot(Rx)
    return R.astype(np.float32)

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

def transform_vertices(R, vts):
    p = np.copy(vts).T
    t = np.mean(p, axis=1).reshape(3, 1)
    pc = p - t
    vts = np.linalg.inv(R).dot(pc)
    vts = vts + t
    return vts.T
#endregion

#region WARP
def GetBilinearPixel(imArr, posX, posY, out):

	#Get integer and fractional parts of numbers
	modXi = int(posX)
	modYi = int(posY)
	modXf = posX - modXi
	modYf = posY - modYi

	#Get pixels in four corners
	for chan in range(imArr.shape[2]):
		bl = imArr[modYi, modXi, chan]
		br = imArr[modYi, modXi+1, chan]
		tl = imArr[modYi+1, modXi, chan]
		tr = imArr[modYi+1, modXi+1, chan]
	
		#Calculate interpolation
		b = modXf * br + (1. - modXf) * bl
		t = modXf * tr + (1. - modXf) * tl
		pxf = modYf * t + (1. - modYf) * b
		out[chan] = int(pxf+0.5) #Do fast rounding to integer

	return None #Helps with profiling view

def WarpProcessing(inArr, outArr, inTriangle, triAffines, shape):

	#Ensure images are 3D arrays
	px = np.empty((inArr.shape[2],), dtype=np.int32)
	homogCoord = np.ones((3,), dtype=np.float32)

	#Calculate ROI in target image
	xmin = shape[:,0].min()
	xmax = shape[:,0].max()
	ymin = shape[:,1].min()
	ymax = shape[:,1].max()
	xmini = int(xmin)
	xmaxi = int(xmax)
	ymini = int(ymin)
	ymaxi = int(ymax)
	#print xmin, xmax, ymin, ymax

	#Synthesis shape norm image		
	for i in range(xmini, xmaxi):
		for j in range(ymini, ymaxi):
			homogCoord[0] = i
			homogCoord[1] = j

			#Determine which tesselation triangle contains each pixel in the shape norm image
			if i < 0 or i >= outArr.shape[1]: continue
			if j < 0 or j >= outArr.shape[0]: continue

			#Determine which triangle the destination pixel occupies
			tri = inTriangle[i,j]
			if tri == -1: 
				continue
				
			#Calculate position in the input image
			affine = triAffines[tri]
			outImgCoord = np.dot(affine, homogCoord)

			#Check destination pixel is within the image
			if outImgCoord[0] < 0 or outImgCoord[0] >= inArr.shape[1]:
				for chan in range(px.shape[0]): outArr[j,i,chan] = 0
				continue
			if outImgCoord[1] < 0 or outImgCoord[1] >= inArr.shape[0]:
				for chan in range(px.shape[0]): outArr[j,i,chan] = 0
				continue

			#Nearest neighbour
			#outImgL[i,j] = inImgL[int(round(inImgCoord[0])),int(round(inImgCoord[1]))]

			#Copy pixel from source to destination by bilinear sampling
			#print i,j,outImgCoord[0:2],im.size
			GetBilinearPixel(inArr, outImgCoord[0], outImgCoord[1], px)
			for chan in range(px.shape[0]):
				outArr[j,i,chan] = px[chan]
			#print outImgL[i,j]

	return None

def PiecewiseAffineTransform(srcIm, srcPoints, dstPoints):
    #Convert input to correct types

	#Split input shape into mesh
    tess = spatial.Delaunay(dstPoints)

	#Calculate ROI in target image
    xmin, xmax = dstPoints[:,0].min(), dstPoints[:,0].max()
    ymin, ymax = dstPoints[:,1].min(), dstPoints[:,1].max()
	#print xmin, xmax, ymin, ymax

    #Determine which tesselation triangle contains each pixel in the shape norm image
    inTessTriangle = np.ones((srcIm.shape[0],srcIm.shape[1]), dtype=np.int) * -1
    for i in range(int(xmin), int(xmax+1.)):
        for j in range(int(ymin), int(ymax+1.)):
            if i < 0 or i >= inTessTriangle.shape[0]:
                continue
            if j < 0 or j >= inTessTriangle.shape[1]:
                continue
            normSpaceCoord = (float(i),float(j))
            simp = tess.find_simplex([normSpaceCoord])
            inTessTriangle[i,j] = simp

	#Find affine mapping from input positions to mean shape
    triAffines = []
    for i, tri in enumerate(tess.vertices):
        meanVertPos = np.hstack((srcPoints[tri], np.ones((3,1)))).transpose()
        shapeVertPos = np.hstack((dstPoints[tri,:], np.ones((3,1)))).transpose()
        affine = np.dot(meanVertPos, np.linalg.inv(shapeVertPos))
        triAffines.append(affine)

	#Prepare arrays, check they are 3D	
    targetArr = np.copy(srcIm)
    srcIm = srcIm.reshape(srcIm.shape[0], srcIm.shape[1], srcIm.shape[2])

    targetArr = targetArr.reshape(targetArr.shape[0], targetArr.shape[1], srcIm.shape[2])

	#Calculate pixel colours
    WarpProcessing(srcIm, targetArr, inTessTriangle, triAffines, dstPoints)
	
	#Convert single channel images to 2D
    if targetArr.shape[2] == 1:
        targetArr = targetArr.reshape((targetArr.shape[0],targetArr.shape[1]))
    return targetArr
#endregion

def warpPerspective(image, rect):
    (tl, tr, br, bl) = rect

    widthA      = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB      = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth    = int(widthA) if int(widthA) > int(widthB) else int(widthB)

    heightA     = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB     = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight   = int(heightA) if int(heightA) > int(heightB) else int(heightB)

    dst = np.array([
        [           0,              0],
        [maxWidth - 1,              0],
        [maxWidth - 1,  maxHeight - 1],
        [           0,  maxHeight - 1]], dtype = "float32")
    P = np.array([
        [      -tl[0],      -tl[1],         -1,          0,          0,      0, tl[0]*dst[0][0], tl[1]*dst[0][0],    dst[0][0]],
        [           0,           0,          0,     -tl[0],     -tl[1],     -1, tl[0]*dst[0][1], tl[1]*dst[0][1],    dst[0][1]],
        [      -tr[0],      -tr[1],         -1,          0,          0,      0, tr[0]*dst[1][0], tr[1]*dst[1][0],    dst[1][0]],
        [           0,           0,          0,     -tr[0],     -tr[1],     -1, tr[0]*dst[1][1], tr[1]*dst[1][1],    dst[1][1]],
        [      -br[0],      -br[1],         -1,          0,          0,      0, br[0]*dst[2][0], br[1]*dst[2][0],    dst[2][0]],
        [           0,           0,          0,     -br[0],     -br[1],     -1, br[0]*dst[2][1], br[1]*dst[2][1],    dst[2][1]],
        [      -bl[0],      -bl[1],         -1,          0,          0,      0, bl[0]*dst[3][0], bl[1]*dst[3][0],    dst[3][0]],
        [           0,           0,          0,     -bl[0],     -bl[1],     -1, bl[0]*dst[3][1], bl[1]*dst[3][1],    dst[3][1]],
        [           0,           0,          0,          0,          0,      0,               0,               0,           1]], dtype = "float32")
    arr_01 = np.array([
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [1]], dtype = "float32")
    P_1 = np.linalg.inv(P)
    H = P_1.dot(arr_01)
    M = np.array([
        [           H[0],              H[1],            H[2]],
        [           H[3],              H[4],            H[5]],
        [           H[6],              H[7],            H[8]]], dtype = "float32")
    M_imp = np.array([
        [           H[0][0],              H[1][0],            H[2][0]],
        [           H[3][0],              H[4][0],            H[5][0]],
        [           H[6][0],              H[7][0],            H[8][0]]], dtype = "float32")
    warped = cv2.warpPerspective(image, M, (resolution, resolution))
    return warped

def flip_texture(tex, isPoseLeft=True):
    X = tex.shape[0]
    Y = tex.shape[1]
    new_tex = np.empty_like(tex)
    if isPoseLeft:
        for y in range(Y//2):
            for x in range(X):
                new_tex[x,Y - y - 1] = tex[x,Y - y - 1]
                new_tex[x,y]         = tex[x,Y - y - 1]
    else:
        for y in range(Y//2):
            for x in range(X):
                new_tex[x,y]         = tex[x,y]
                new_tex[x,Y - y - 1] = tex[x,y]
    return new_tex

def create_scatter(img, vts, kpt, isMesh=False):
    img = cv2.resize(img, (256,256))
    if isMesh:
        x, y, z = vts.transpose()
        for i in range(0, x.shape[0], 1):
            img = cv2.circle(img, (int(x[i]), int(y[i])), 1, (255, 0, 0), -1)
    x, y, z = kpt.transpose().astype(np.int32)
    for i in range(0, x.shape[0], 1):
        if i in face_contour_ind:
            img = cv2.circle(img, (int(x[i]), int(y[i])), 5, (255, 0, 0), -1)
        else:
            img = cv2.circle(img, (int(x[i]), int(y[i])), 5, (255, 255, 255), -1)
    return img

def show_result(*img, columns, rows):
    fig=plt.figure(figsize=(8, 8))
    for i in range(1, columns*rows +1):
        show_img = img[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(show_img)
    plt.savefig('FaceRotation_Demo.png')
    plt.show()
# Set profiling angle
phi_delta   = 20/180*math.pi
gamma_delta = 10/180*math.pi
theta_delta = 0
# Load Sample
working_folder      = str(os.path.abspath(os.getcwd()))
img                 = cv2.imread(os.path.join(working_folder, "result/store", "image00013.jpg"))
img                 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pos                 = np.load(os.path.join(working_folder, "result/store", "image00013.npy")).astype(np.float32)
tex                 = cv2.remap(img, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
# Load Model
M_face_contour      = loadmat(os.path.join(working_folder, 'test.synface/Model_face_contour_trimed.mat'))
M_fullmod_contour   = loadmat(os.path.join(working_folder, 'test.synface/Model_fullmod_contour.mat'))
M_tri_mouth         = loadmat(os.path.join(working_folder, 'test.synface/Model_tri_mouth.mat'))
M_keypoints         = loadmat(os.path.join(working_folder, 'test.synface/Model_keypoints.mat'))
tri                 = np.loadtxt(os.path.join(working_folder, 'test.synface/triangles.txt')).astype(np.int32)
tri_plus            = np.concatenate((tri, np.asarray(M_tri_mouth["tri_mouth"], dtype=np.int32).T))
layer_width         = [0.1, 0.15, 0.2, 0.25, 0.3]

FLAGS = {   
            "model"             : os.path.join(working_folder, "train_log/_checkpoint_epoch_80.pth.tar"),
            "data_path"         : os.path.join(working_folder, "data"),
			"uv_kpt_ind_path"   : os.path.join(working_folder, "data/processing/Data/UV/uv_kpt_ind.txt"),
            "face_ind_path"     : os.path.join(working_folder, "data/processing/Data/UV/face_ind.txt"),
            "triangles_path"    : os.path.join(working_folder, "data/processing/Data/UV/triangles.txt"),
            "canonical_vts_path": os.path.join(working_folder, "data/processing/Data/UV/canonical_vertices.npy"),
			"result_path"       : os.path.join(working_folder, "result/usr"),
            "uv_kpt_path"       : os.path.join(working_folder, "data/processing/Data/UV/uv_kpt_ind.txt"),
            "device"            : "cuda",
            "devices_id"        : [0],
            "batch_size"        : 16, 
            "workers"           : 8
		}

uv_kpt_ind          = np.loadtxt(FLAGS["uv_kpt_ind_path"]).astype(np.int32)
kpt_ind             = np.array([8444, 8529, 8702, 8763, 9168, 9203, 9246, 9281, 10877, 11016, 13407, 13611, 13694, 13866, 13931, 14857, 14908, 15325, 15424, 15589, 15652, 15826, 15907, 16851, 20049, 22396, 22509, 22621, 26188, 26209, 26682, 26693, 27175, 27792, 28003, 30014, 30021, 30250, 30926, 30949, 31618, 31838, 31849, 33074, 33158, 33277, 33375, 33395, 33412, 33413, 33608, 33617, 34680, 34699, 35110, 35115, 35119, 38077, 38268, 41382, 41547, 42101, 42132, 42234, 42307, 42506, 42627, 42986], dtype=np.int)
face_ind            = np.loadtxt(FLAGS["face_ind_path"]).astype(np.int32)
triangles           = np.loadtxt(FLAGS["triangles_path"]).astype(np.int32)
canonical_vertices  = np.load(FLAGS["canonical_vts_path"])
resolution  = 256
# Construct 3D Face
vts                 	= get_vertices(pos)
clr 					= get_vertices(tex)
kpt                     = get_landmarks(pos)
face_contour_ind        = list(range(0, 28))
sct_img                 = create_scatter(img, vts, kpt)
ren_img                 = render_colors(vts, triangles, clr, resolution, resolution).astype(np.uint8)

P, pose, (s, R, t) 		= estimate_pose(vts)
### pose (0.336254458754315, -0.032371088523203216, -0.3484050027460824)
new_pose                = (phi_delta, gamma_delta, theta_delta)
rot_vts                 = transform_vertices(angle2matrix(new_pose), vts)
rot_pos                 = transform_vertices(angle2matrix(new_pose), np.reshape(pos, [resolution**2, -1]))
rot_pos                 = np.reshape(rot_pos, [resolution, resolution, -1])
# inp_tex               = flip_texture(tex, isPoseLeft=True)
# rot_clr 				= get_vertices(inp_tex)
rot_img                 = render_colors(rot_vts, triangles, clr, resolution, resolution).astype(np.uint8)

# brg_points               = np.array([
#                                         [0,             0,   1], [resolution//2, 0, 1], [resolution - 1, 0, 1],
#                                         [0, resolution//2,   1], [resolution - 1, resolution//2, 1],
#                                         [0, resolution -1,   1], [resolution//2, resolution -1, 1], [resolution - 1, resolution -1, 1]
#                                     ], dtype=np.float32)
brg_points               = np.array([
                                        [             0,             0,   1], [resolution - 1,             0,  1],
                                        [resolution - 1, resolution -1,   1], [0             , resolution -1,  1]
                                    ], dtype=np.float32)
background               = np.zeros_like(img) #Need to find a method to warp the background follow the face
rot_img                  = np.where(np.sum(rot_img, axis=2, keepdims=True) > 0, rot_img, background)
# show_result(img, sct_img, ren_img, rot_img, columns=2, rows=2)
# Image Meshing
# texture             = cv2.remap(img, uv_position_map[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
# np.save(os.path.join("/home/viet/Projects/Pycharm/SPRNet/result/usr", "image1_texture.jpg"), tex)

clean_tex = cv2.remap(ren_img, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
miss_tex = cv2.remap(rot_img, rot_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
scipy.misc.imsave(os.path.join("/home/viet/Projects/Pycharm/SPRNet/result/usr", "image1_texture.jpg"), clean_tex)
scipy.misc.imsave(os.path.join("/home/viet/Projects/Pycharm/SPRNet/result/usr", "image1_misstexture.jpg"), miss_tex)
# Rotating and Anchor Adjustment

# Get Rotating Result


