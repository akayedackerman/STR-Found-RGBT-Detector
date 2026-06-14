import os.path as osp
import mmcv
import numpy as np

from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadThermalImageFromFile(BaseTransform):

	def transform(self, results):
		rgb_path = results['img_path']

		# Replace visible folder with thermal folder
		thermal_path = rgb_path.replace('/00/', '/01/')

		# Load thermal image as grayscale
		thermal = mmcv.imread(
			thermal_path,
			flag='grayscale'
		)

		if thermal is None:
			raise FileNotFoundError(
				f'Failed to load thermal image: {thermal_path}'
			)

		# Convert to float32
		thermal = thermal.astype(np.float32)

		# Normalize to [0,1]
		thermal = thermal / 255.0

		# Expand to single channel
		thermal = np.expand_dims(
			thermal,
			axis=-1
		)

		# Replicate to 3 channels
		thermal = np.concatenate(
			[thermal, thermal, thermal],
			axis=-1
		)

		# Safety checks
		if np.isnan(thermal).any():
			raise ValueError(
				f'NaN detected in thermal image: {thermal_path}'
			)

		if np.isinf(thermal).any():
			raise ValueError(
				f'Inf detected in thermal image: {thermal_path}'
			)

		results['thermal_img'] = thermal

		return results


@TRANSFORMS.register_module()
class ConcatThermalToRGB(BaseTransform):

	def transform(self, results):

		rgb = results['img'].astype(np.float32)
		thermal = results['thermal_img'].astype(np.float32)

		# BGR -> RGB
		rgb = rgb[..., [2, 1, 0]]

		# Normalize RGB to [0,1]
		rgb = rgb / 255.0

		# Shape validation
		if rgb.shape[:2] != thermal.shape[:2]:
			raise ValueError(
				f'RGB/Thermal shape mismatch: '
				f'{rgb.shape} vs {thermal.shape}'
			)

		# RGB normalization
		mean_rgb = np.array(
			[0.485, 0.456, 0.406],
			dtype=np.float32
		)

		std_rgb = np.array(
			[0.229, 0.224, 0.225],
			dtype=np.float32
		)

		rgb = (rgb - mean_rgb) / std_rgb

		# Thermal normalization
		thermal = (thermal - 0.5) / 0.5

		# Concatenate RGB + Thermal
		rgbt = np.concatenate(
			[rgb, thermal],
			axis=-1
		)

		# Final safety checks
		if np.isnan(rgbt).any():
			raise ValueError('NaN detected in RGBT tensor')

		if np.isinf(rgbt).any():
			raise ValueError('Inf detected in RGBT tensor')

		results['img'] = rgbt

		return results


@TRANSFORMS.register_module()
class LoadTemporalRGBTPair(BaseTransform):
	"""Spatio-Temporal Frame-Pairing Pipeline Component for STR-Found.

    Loads Frame(t) and Frame(t-1), executes your exact multi-modal
    normalization values, and returns a unified [H, W, 12] tensor array.
    """

	def __init__(self):
		super().__init__()
		self.mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32)
		self.std_rgb = np.array([0.229, 0.224, 0.225], dtype=np.float32)

	def _process_single_frame(self, rgb_path, thermal_path, results_raw_img=None):
		# Read items from system storage disk space
		if results_raw_img is not None and osp.exists(rgb_path):
			# If checking current frame, reuse image matrix already cached in mmcv results pipeline
			rgb_raw = results_raw_img.astype(np.float32)
		else:
			if not osp.exists(rgb_path):
				raise FileNotFoundError(f"Sequence tracking frame missing path location target: {rgb_path}")
			rgb_raw = mmcv.imread(rgb_path).astype(np.float32)

		thermal_raw = mmcv.imread(thermal_path, flag='grayscale')
		if thermal_raw is None:
			raise FileNotFoundError(f"Sequence tracking infrared missing path location target: {thermal_path}")

		thermal_raw = thermal_raw.astype(np.float32)

		# 1. BGR -> RGB and normalization mapping tracking
		rgb = rgb_raw[..., [2, 1, 0]] / 255.0
		rgb = (rgb - self.mean_rgb) / self.std_rgb

		# 2. Thermal transformations mapping tracking
		thermal = thermal_raw / 255.0
		thermal = np.expand_dims(thermal, axis=-1)
		thermal = np.concatenate([thermal, thermal, thermal], axis=-1)
		thermal = (thermal - 0.5) / 0.5

		# 3. Structural shape validation test
		if rgb.shape[:2] != thermal.shape[:2]:
			raise ValueError(f"Spatial alignment layout mismatch error: {rgb.shape} vs {thermal.shape}")

		return np.concatenate([rgb, thermal], axis=-1)

	def transform(self, results):
		# Trace down sequence index indexes from current Frame(t) details
		curr_rgb_path = results['img_path']
		curr_thermal_path = curr_rgb_path.replace('/00/', '/01/')

		base_dir, file_name = osp.split(curr_rgb_path)
		name_part, ext_part = osp.splitext(file_name)

		try:
			frame_idx = int(name_part)
			# Find index location for historical state data frame (t-1)
			prev_idx = max(0, frame_idx - 1)
			prev_rgb_path = osp.join(base_dir, f"{prev_idx:06d}{ext_part}")
		except ValueError:
			# Fallback path if index formatting structures alter dynamically
			prev_rgb_path = curr_rgb_path

		prev_thermal_path = prev_rgb_path.replace('/00/', '/01/')

		# Enforce boundary thresholds safely (if past frame does not exist, pad with itself)
		if not osp.exists(prev_rgb_path) or not osp.exists(prev_thermal_path):
			prev_rgb_path = curr_rgb_path
			prev_thermal_path = curr_thermal_path

		# Generate individual processed multi-channel tensors
		rgbt_curr = self._process_single_frame(curr_rgb_path, curr_thermal_path, results.get('img'))
		rgbt_prev = self._process_single_frame(prev_rgb_path, prev_thermal_path, None)

		# Combine into complete 12-channel configuration array matrix block
		fused_12ch_tensor = np.concatenate([rgbt_curr, rgbt_prev], axis=-1)

		# Execution safety assertions check validation target parameters
		if np.isnan(fused_12ch_tensor).any():
			raise ValueError('NaN artifact anomaly values captured inside fused temporal 12-channel tensor array.')
		if np.isinf(fused_12ch_tensor).any():
			raise ValueError('Inf artifact anomaly values captured inside fused temporal 12-channel tensor array.')

		results['img'] = fused_12ch_tensor
		results['img_shape'] = fused_12ch_tensor.shape[:2]
		results['ori_shape'] = fused_12ch_tensor.shape[:2]
		return results