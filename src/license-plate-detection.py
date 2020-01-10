import sys, os
import keras
import cv2
import traceback

from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes

import numpy as np 
import matplotlib.pyplot as plt 
import time
def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


if __name__ == '__main__':

	try:
		
		#input_dir  = sys.argv[1]
		input_dir = "E:\\project\\test8"
		output_dir = input_dir

		lp_threshold = 0.5

		#wpod_net_path = sys.argv[2]
		wpod_net_path = "wpod-net_update1.h5"
		wpod_net = load_model(wpod_net_path)

		imgs_paths = glob('%s/*.png' % input_dir)

		print('Searching for license plates using WPOD-NET')

		for i,img_path in enumerate(imgs_paths):
			start = time.time()
			print('\t Processing %s' % img_path)

			bname = splitext(basename(img_path))[0]
			Ivehicle = cv2.imread(img_path)
			ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
			side  = int(ratio*288.) #288
			bound_dim = min(side + (side%(2**4)),608) #608
			print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

			ratio_w_h = Ivehicle.shape[1]/Ivehicle.shape[0]
			Llp, LlpImgs = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(300,220),lp_threshold, ratio_w_h)

			if len(LlpImgs):
				Ilp = LlpImgs[0]
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

				s = Shape(Llp[0].pts)	
				x1, y1 = int(np.amin(Llp[0].pts, 1)[0]*Ivehicle.shape[1]), int(np.amin(Llp[0].pts, 1)[1]*Ivehicle.shape[0])
				x2, y2 = int(np.amax(Llp[0].pts, 1)[0]*Ivehicle.shape[1]), int(np.amax(Llp[0].pts, 1)[1]*Ivehicle.shape[0])
				print(x1, y1, x2, y2)
				#cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
				#writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
			
				cv2.rectangle(Ivehicle, (x1, y1), (x2, y2),(0,255,0), 2)

				print("timing: ", time.time() - start)
				#plt.imshow((Ilp*255).astype(np.uint8))
				plt.imshow(Ivehicle)
				plt.show()
			
	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)


