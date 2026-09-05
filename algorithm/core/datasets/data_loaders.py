import os
import rasterio
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def read_img_names_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        return [line.strip() + '.tif' for line in file if line.strip()]


class GUM(Dataset):
    def __init__(self, img_dirs, txt_paths, transform=None, inference_mode=False,
                 label_dir=None):
        assert len(img_dirs) == len(txt_paths)
        self.transform = transform
        self.inference_mode = inference_mode
        self.label_dir = label_dir

        self.data = []
        for img_dir, txt_path in zip(img_dirs, txt_paths):
            if txt_path != 'None':
                img_filenames = sorted(set(read_img_names_from_txt(txt_path)))
            else:
                img_filenames = sorted(os.listdir(img_dir))
            for img_filename in tqdm(img_filenames):
                img_path = os.path.join(img_dir, img_filename)
                label_path = os.path.join(self.label_dir, img_filename)
                self.data.append((img_path, label_path))

        print('There are', len(self.data), 'files!')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_path = self.data[idx]

        with rasterio.open(img_path) as data:
            multimodal_data = data.read()

        if not self.inference_mode:
            with rasterio.open(label_path) as data:
                label = data.read()
            if label.shape != (1, 256, 256):
                print(label_path)

        multimodal_data = self._prepare_multimodal(multimodal_data)

        img = np.transpose(multimodal_data, (1, 2, 0))

        if not self.inference_mode:
            label = np.transpose(label, (1, 2, 0)).astype(np.int64)

        if self.transform:
            img = self.transform(img)
            if not self.inference_mode:
                label = self.transform(label)

        file_name = os.path.basename(img_path)
        img_meta = {
            'filename': file_name,
            'ori_filename': file_name,
            'img': img,
            'img_shape': img.shape,
            'ori_shape': img.shape,
            'pad_shape': img.shape,
            'scale_factor': 1.0,
            'flip': False,
        }
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        img_meta['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        if self.inference_mode:
            return img, img_meta
        return img, label, img_meta

    def _prepare_multimodal(self, data):
        """Normalize S2, S1, and topography channels in place."""
        # S2: 4 optical bands, already expected in [0, 1].
        data[0:4] = _normalize_scalar(data[0:4], vmin=0.0, vmax=1.0, nan=0.0)

        # S1: 4 SAR bands, clipped to per-band 0.5-99.5% thresholds.
        s1_clip_min = np.array([-23.62251091003418, -37.2230110168457,
                                -24.20653533935547, -37.36018753051758], dtype=np.float32)
        s1_clip_max = np.array([2.138315200805664, 0.0,
                                2.071335747242074, 0.0], dtype=np.float32)
        data[4:8] = self._normalize_per_channel(data[4:8], s1_clip_min, s1_clip_max)

        # Topography: channel 8 is raw DEM height (unused as input).
        # Normalize slope/aspect from channels 9-10 and store them in channels 8-9
        # so the final 10-channel input is [S2 x4, S1 x4, slope, aspect].
        slope = _normalize_scalar(data[9:10], vmin=0.0, vmax=90.0, nan=0.0)
        aspect = _normalize_scalar(data[10:11], vmin=0.0, vmax=360.0, nan=0.0)
        data[8:10] = np.concatenate((slope, aspect), axis=0)

        return data[:10]

    @staticmethod
    def _normalize_per_channel(data, vmin, vmax):
        """Replace NaNs/inf per channel, clip, and min-max normalize to [0, 1]."""
        data = data.astype(np.float32)
        for i in range(data.shape[0]):
            data[i] = np.nan_to_num(
                data[i], nan=vmax[i], posinf=vmax[i], neginf=vmin[i])
        vmin = vmin.reshape(-1, 1, 1)
        vmax = vmax.reshape(-1, 1, 1)
        data = np.clip(data, vmin, vmax)
        return (data - vmin) / (vmax - vmin)


def _normalize_scalar(data, vmin, vmax, nan):
    """Replace NaNs/inf, clip, and min-max normalize to [0, 1]."""
    data = np.nan_to_num(data, nan=nan, posinf=nan, neginf=vmin)
    return np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
