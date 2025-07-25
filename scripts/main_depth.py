import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off different numerical values due to rounding errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # enables more tf instructions in operations
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline
import cv2
import numpy as np
import tensorflow as tf
import vpi
import time
from threading import Thread
from datetime import datetime




# All the connections between keypoints
EDGES = {
	(0, 1): 'm',
	(0, 2): 'c',
	(1, 3): 'm',
	(2, 4): 'c',
	(0, 5): 'm',
	(0, 6): 'c',
	(5, 7): 'm',
	(7, 9): 'm',
	(6, 8): 'c',
	(8, 10): 'c',
	(5, 6): 'y',
	(5, 11): 'm',
	(6, 12): 'c',
	(11, 12): 'y',
	(11, 13): 'm',
	(13, 15): 'm',
	(12, 14): 'c',
	(14, 16): 'c',
}

# Variables
edge_color =  (0,0,255) #Red
point_color = (0,255,0) #Green

# Draw the keypoints as circles in the frame
def draw_keypoints(frame, keypoints, confidence_threshold):
	y, x, c = frame.shape
	shaped_array = np.squeeze(np.multiply(keypoints, [y,x,1]))
	for kp in shaped_array:
		ky, kx, kp_conf = kp
		if kp_conf > confidence_threshold:
			cv2.circle(frame, (int(kx), int(ky)), 4, point_color, -1)


# Draw the edges between the coordinates
def draw_connections(frame, keypoints, edges, confidence_threshold):
	y, x, c = frame.shape
	shaped_array = np.squeeze(np.multiply(keypoints, [y,x,1]))
	for edge, color in edges.items(): 
		p1, p2 = edge
		y1, x1, c1 = shaped_array[p1]
		y2, x2, c2 = shaped_array[p2]
		if (c1 > confidence_threshold) & (c2 > confidence_threshold):
			cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), edge_color, 2)


# Initialize left and right CSI cameras using GStreamer
class CameraThread(Thread):
	def __init__(self, sensor_id) -> None:
		super().__init__()
		self._cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id), cv2.CAP_GSTREAMER)
		self._should_run = True
		self._image = None
		self.start()
	def run(self):
		while self._should_run:
			ret,frame = self._cap.read()
			if ret:
				self._image = frame
	@property
	def image(self):
		# NOTE: if we care about atomicity of reads, we can add a lock here
		return self._image
	def stop(self):
		self._should_run = False
		self._cap.release()


def get_calibration() -> tuple:
	data = np.load("params/disp_params_rectified.npz")
	map_l = (data["map1_l"], data["map2_l"])
	map_r = (data["map1_r"], data["map2_r"])
	return map_l, map_r


MAX_DISP = 128
WINDOW_SIZE	= 10





if __name__ == '__main__':
	
	# Loads the model from the file.
	interpreter = tf.lite.Interpreter(model_path='models/movenet-thunder.tflite')
	interpreter.allocate_tensors()
	
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	
	map_l, map_r = get_calibration()
	cam_l = CameraThread(0)
	cam_r = CameraThread(1)
	
	#time.sleep(0.5)
	#for _ in range(5):
		#_ = cam_l.image
		#_ = cam_r.image
		#time.sleep(0.05)
	#frame_l = cam_l.image
	#frame_r = cam_r.image

	print("Waiting for the first valid frames from the cameras...")
	while cam_l.image is None or cam_r.image is None:
    		time.sleep(0.01)
	print("Frames received. Starting the application.")
	
	try:
		with vpi.Backend.CUDA:
			while True:
				arr_l = cam_l.image
				arr_r = cam_r.image
				#for _ in range(5):
					#_ = cam_l.image
					#_ = cam_r.image
					#time.sleep(0.05)
				
				# RGB -> GRAY
				arr_l = cv2.cvtColor(arr_l, cv2.COLOR_BGR2GRAY)
				arr_r = cv2.cvtColor(arr_r, cv2.COLOR_BGR2GRAY)
				
				# Rectify
				arr_l_rect = cv2.remap(arr_l, *map_l, cv2.INTER_LANCZOS4)
				arr_r_rect = cv2.remap(arr_r, *map_r, cv2.INTER_LANCZOS4)
				
				# Apply slight Gaussian blur to reduce high-frequency noise
				arr_l_rect = cv2.GaussianBlur(arr_l_rect, (3, 3), 0)
				arr_r_rect = cv2.GaussianBlur(arr_r_rect, (3, 3), 0)
				
				# Resize
				arr_l_rect = cv2.resize(arr_l_rect, (480, 270))
				arr_r_rect = cv2.resize(arr_r_rect, (480, 270))
				
				# Convert to VPI image
				vpi_l = vpi.asimage(arr_l_rect)
				vpi_r = vpi.asimage(arr_r_rect)
				
				vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
				vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)
				
				disparity_16bpp = vpi.stereodisp(
					vpi_l_16bpp,
					vpi_r_16bpp,
					out_confmap = None,
					backend = vpi.Backend.CUDA,
					window = WINDOW_SIZE,
					maxdisp = MAX_DISP,
				)
				
				disp_raw = disparity_16bpp.convert(vpi.Format.U8, scale=255.0 / (32 * MAX_DISP)).cpu().copy()
				disp_vis = cv2.medianBlur(disp_raw, 5)
				disp_arr = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
				
				# Prepare input for MoveNet from left rectified frame
				frame_rgb = cv2.cvtColor(arr_l_rect, cv2.COLOR_GRAY2BGR)
				input_image = tf.image.resize_with_pad(np.expand_dims(frame_rgb, axis=0), 256, 256)
				input_image = tf.cast(input_image, dtype=tf.float32) / 255.0
				
				# Run MoveNet inference
				interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
				interpreter.invoke()
				keypoints = np.squeeze(interpreter.get_tensor(output_details[0]['index']))

				draw_img = disp_arr.copy()
				
				# Depth estimation parameters
				baseline = 0.1		# Distance between cameras in meters
				focal_length = 580	# Focal length in pixels (from calibration)

				cx = 240
				cy = 135
				
				# Image size for keypoint mapping
				h, w = frame_rgb.shape[:2]
				
				# Iterate over each keypoint to calculate depth from disparity
				for y_norm, x_norm, conf in keypoints:
					if conf < 0.4:
						continue
					x = int(x_norm * w)
					y = int(y_norm * h)
					if 0 <= x < w and 0 <= y < h:
						disparity_val = disp_raw[y, x]  # Use raw disparity value (single channel)
						if disparity_val > 0:
							z = (baseline * focal_length) / disparity_val
							X = (x - cx) * z / focal_length
							Y = (y - cy) * z / focal_length
							label = f"x:{X:.2f}m, y:{Y:.2f}m, z:{z:.2f}m"
							cv2.circle(draw_img, (x, y), 4, (0, 255, 0), -1)
							cv2.putText(draw_img, label, (x + 5, y - 10),
							        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
				
				# Draw skeleton on frame
				draw_connections(draw_img, keypoints, EDGES, 0.4)
				draw_keypoints(draw_img, keypoints, 0.4)
				
				# Resize outputs for display
				disp_show = cv2.resize(disp_arr, (640, 360))
				pose_show = cv2.resize(frame_rgb, (640, 360))
				
				# Show both disparity and pose estimation results
				cv2.imshow("Disparity", disp_show)
				cv2.imshow("Pose Estimation", pose_show)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				
	except KeyboardInterrupt as e:
		print(e)
	finally:
		cam_l.stop()
		cam_r.stop()
		cv2.destroyAllWindows()

