
import numpy as np
import cv2
import time

from os.path import splitext

from src.label import Label
from src.utils import getWH, nms
from src.projection_utils import getRectPts, find_T_matrix
import matplotlib.pyplot as plt 

class DLabel (Label):

	def __init__(self, cl, pts, prob, ratio):
		# if (pts[0][1] - pts[0][0])*640/(pts[1][2]-pts[1][0])/480 < 2: # bien so vuong
		# 	pts[0] = pts[0] + np.array([-5,5,5,-5])/640.   #x
		# 	pts[1] = pts[1] + np.array([-5,-5,5,5])/480.   #y
		# else:
		# 	pts[0] = pts[0] + np.array([-5,5,5,-5])/640.   #x
		# 	w = (pts[0][1] - pts[0][0])
			
		# 	pts[0][0] = pts[0][0] + w*0.35
		# 	pts[0][3] = pts[0][3] + w*0.35
		# 	pts[1] = pts[1] + np.array([-5,-5,5,5])/480.   #y			
		#print("ratio: ", (pts[0][1] - pts[0][0])*640/(pts[1][2]-pts[1][0])/480)

		if (pts[0][1] - pts[0][0])/(pts[1][2]-pts[1][0])*ratio > 2: # bien so dai
			w = (pts[0][1] - pts[0][0])
			pts[0][0] = pts[0][0] + w*0.4
			pts[0][3] = pts[0][3] + w*0.4	
		self.pts = pts
		tl = np.amin(pts,1) 
		br = np.amax(pts,1) 
		Label.__init__(self,cl,tl,br,prob)

def save_model(model,path,verbose=0):
	path = splitext(path)[0]
	model_json = model.to_json()
	with open('%s.json' % path,'w') as json_file:
		json_file.write(model_json)
	model.save_weights('%s.h5' % path)
	if verbose: print('Saved to %s' % path)

def load_model(path,custom_objects={},verbose=0):
	from keras.models import model_from_json

	path = splitext(path)[0]
	with open('%s.json' % path,'r') as json_file:
		model_json = json_file.read()
	model = model_from_json(model_json, custom_objects=custom_objects)
	model.load_weights('%s.h5' % path)
	if verbose: print('Loaded from %s' % path)
	return model


def reconstruct(Iorig,I,Y,out_size, ratio, threshold=.9):

	net_stride 	= 2**4
	side 		= ((208. + 40.)/2.)/net_stride # 7.75

	Probs = Y[...,0]
	Affines = Y[...,2:]
	rx,ry = Y.shape[:2]
	ywh = Y.shape[1::-1]
	iwh = np.array(I.shape[1::-1],dtype=float).reshape((2,1))

	xx,yy = np.where(Probs>threshold)
	WH = getWH(I.shape)
	MN = WH/net_stride

	vxx = vyy = 0.5 #alpha

	base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
	labels = []

	for i in range(len(xx)):
		y,x = xx[i],yy[i]
		affine = Affines[y,x]
		prob = Probs[y,x]

		mn = np.array([float(x) + .5,float(y) + .5])

		A = np.reshape(affine,(2,3))
		A[0,0] = max(A[0,0],0.)
		A[1,1] = max(A[1,1],0.)

		pts = np.array(A*base(vxx,vyy)) #*alpha
		pts_MN_center_mn = pts*side
		pts_MN = pts_MN_center_mn + mn.reshape((2,1))

		pts_prop = pts_MN/MN.reshape((2,1))
		labels.append(DLabel(0,pts_prop,prob, ratio))
	final_labels = nms(labels,.1)
	TLps = []
	if len(final_labels):
		final_labels.sort(key=lambda x: x.prob(), reverse=True)
		for i,label in enumerate(final_labels):
			if (label.pts[0][1] - label.pts[0][0])/(label.pts[1][2]-label.pts[1][0])*ratio <= 2:
				t_ptsh = getRectPts(0,0,out_size[0],out_size[1])
			else:
				t_ptsh = getRectPts(0, 0, 300, 110)
			ptsh = np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)),np.ones((1,4))))
			H = find_T_matrix(ptsh,t_ptsh)
			if (label.pts[0][1] - label.pts[0][0])/(label.pts[1][2]-label.pts[1][0])*ratio <= 2:
				Ilp = cv2.warpPerspective(Iorig, H, out_size, borderValue=.0)
			else:
				Ilp = cv2.warpPerspective(Iorig, H, (300, 110), borderValue=.0)
			TLps.append(Ilp)

	return final_labels,TLps
	

def detect_lp(model, I, max_dim, net_step, out_size, threshold, ratio):

	min_dim_img = min(I.shape[:2])
	factor 		= float(max_dim)/min_dim_img

	w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
	w += (w%net_step!=0)*(net_step - w%net_step)
	h += (h%net_step!=0)*(net_step - h%net_step)
	Iresized = cv2.resize(I,(w,h))

	T = Iresized.copy()
	T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))
	Yr 		= model.predict(T)
	Yr 		= np.squeeze(Yr)
	L,TLps = reconstruct(I, Iresized, Yr, out_size, ratio, threshold)


	# fig, axes = plt.subplots(4, 2)
	# for i in range(4):
	# 	for j in range(2):
	# 		plt.subplot(4,2,(i*2+j + 1))
	# 		plt.imshow(Yr[:,:,i*2+j])
	# plt.show()
	#print(np.where(Yr[:,:,1] < 0.5))
	# print(max(Yr[:,:,0].flatten()))
	# print(np.where(Yr[:,:,0] == max(Yr[:,:,0].flatten())))
	#cY, cX = np.where(Yr[:,:,0] == max(Yr[:,:,0].flatten()))
	return L,TLps