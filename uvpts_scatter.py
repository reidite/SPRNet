# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import os
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm
import scipy.io as sio
from pathlib import Path
import numpy as np
from utils.toolkit import *
import cv2
import math
import scipy.spatial as spatial
from utils.toolkit import *
import matplotlib.pyplot as plt
from skimage import measure
from PIL import Image
def plot_mesh(vertices, triangles, subplot = [1,1,1], title = '', el = 100, az = 80, lwdt=-1.3, dist = 6, color = "red"):
    ax = plt.subplot(subplot[0], subplot[1], subplot[2], projection = '3d')
	# ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles = triangles, lw = lwdt, color = color, alpha = 1)
    downsize = 10
    ax.scatter(vertices[::downsize, 0], vertices[::downsize, 1], vertices[::downsize, 2], c=color, marker='o')
    ax.axis("off")
    ax.view_init(elev = el, azim = az)
    ax.dist = dist
    plt.title(title)
    plt.show()

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

def WarpProcessing(inIm, inArr, outArr, inTriangle, triAffines, shape):

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

def PiecewiseAffineTransform(srcIm, srcPoints, dstIm, dstPoints):

	#Convert input to correct types
	srcArr = np.asarray(srcIm, dtype=np.float32)
	dstPoints = np.array(dstPoints)
	srcPoints = np.array(srcPoints)

	#Split input shape into mesh
	tess = spatial.Delaunay(dstPoints)

	#Calculate ROI in target image
	xmin, xmax = dstPoints[:,0].min(), dstPoints[:,0].max()
	ymin, ymax = dstPoints[:,1].min(), dstPoints[:,1].max()
	#print xmin, xmax, ymin, ymax

	#Determine which tesselation triangle contains each pixel in the shape norm image
	inTessTriangle = np.ones(dstIm.size, dtype=np.int) * -1
	for i in range(int(xmin), int(xmax+1.)):
		for j in range(int(ymin), int(ymax+1.)):
			if i < 0 or i >= inTessTriangle.shape[0]: continue
			if j < 0 or j >= inTessTriangle.shape[1]: continue
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
	targetArr = np.copy(np.asarray(dstIm, dtype=np.uint8))
	srcArr = srcArr.reshape(srcArr.shape[0], srcArr.shape[1], len(srcIm.mode))
	targetArr = targetArr.reshape(targetArr.shape[0], targetArr.shape[1], len(dstIm.mode))

	#Calculate pixel colours
	WarpProcessing(srcIm, srcArr, targetArr, inTessTriangle, triAffines, dstPoints)
	
	#Convert single channel images to 2D
	if targetArr.shape[2] == 1:
		targetArr = targetArr.reshape((targetArr.shape[0],targetArr.shape[1]))
	dstIm.paste(Image.fromarray(targetArr))


def transform_vertices(R, vts):
    p = np.copy(vts).T
    t = np.mean(p, axis=1).reshape(3, 1)
    pc = p - t
    vts = np.linalg.inv(R).dot(pc)
    vts = vts + t
    return vts.T




if __name__ == "__main__":
    # file_path           = (str(os.path.abspath(os.getcwd())))
    # data_list_val       = os.path.join(file_path, "test.configs", "AFLW2000-3D.list")
    # img_names_list      = Path(data_list_val).read_text().strip().split('\n')
    # data_index          = 123
    # file_name           = os.path.splitext(img_names_list[data_index])[0]
    # uv_position_map     = np.load(os.path.join(file_path, "data", "verify_uv_256x256", file_name + ".npy")).astype(np.float32)
    # kpt                 = get_landmarks(uv_position_map)
    # img                 = cv2.imread(os.path.join(file_path, "data", "verify_im_256x256", file_name + ".jpg"))
    # show_uv_mesh(img, uv_position_map, kpt, False)
    img                 = cv2.imread("/home/viet/Projects/Pycharm/SPRNet/result/store/image00008.jpg")
    img                 = (img / 255.0).astype(np.float32)
    
    uv_position_map     = np.load(os.path.join("/home/viet/Projects/Pycharm/SPRNet/result/store", "image00008.npy")).astype(np.float32)
    kpt                 = get_landmarks(uv_position_map)
    show_uv_mesh(img, uv_position_map, kpt, False)
    # srcIm = Image.open("/home/viet/Projects/Pycharm/SPRNet/result/store/image00008.jpg")	
    # dstIm = Image.new(srcIm.mode,(500,500))

    
    # texture             = cv2.remap(img, uv_position_map[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    # np.save(os.path.join("/home/viet/Projects/Pycharm/SPRNet/result/usr", "image1_texture.npy"), texture)
    ### Create Large Pose
	# srcCloud            = get_vertices(uv_position_map)
	# camera_matrix, pose, (s, R, t) = estimate_pose(srcCloud)
	# R                       = angle2matrix(angle)
	# dstCloud               = transform_vertices(R, srcCloud)
    
    
    # kpt                 = get_landmarks(uv_position_map)
    # show_uv_mesh(img, uv_position_map, kpt, False)
    texture             	= cv2.remap(img, uv_position_map[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    vts                 	= get_vertices(uv_position_map)
    tex 					= get_vertices(texture)
    P, pose, (s, R, t) 		= estimate_pose(vts)
    front_vts               = transform_vertices(R, vts) * 2
    # plot_mesh(vts, triangles)
    # texcoord = np.zeros_like(uv_position_map) 
    # texcoord[:,0] = uv_position_map[:,0]*(256 - 1)
    # texcoord[:,1] = uv_position_map[:,1]*(256 - 1)
    # texcoord[:,1] = 256 - texcoord[:,1] - 1
    # texcoord = np.hstack((texcoord, np.zeros((texcoord.shape[0], 1))))
    # img_show    			= render_texture(vts, triangles, texture, texcoord, triangles,256, 256, 3)
    # img_show                = cv2.resize(img_show, None, fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
    # for i in range(0, front_vts.shape[0], 1):
    # 	img_show = cv2.circle(img_show, (int(front_vts[i][0]), int(front_vts[i][1])), 1, (250, 0, 0), -1)
    
    # print("Success")
