import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from lib import dataset
from lib import nets
from lib import spec_utils
from lib import utils

from datetime import datetime
start_time = datetime.now()
import uuid
from cog import BasePredictor, Input, Path, BaseModel
from typing import List

class Separator(object):

    def __init__(self, model, device, batchsize, cropsize, postprocess=False):
        self.model = model
        self.offset = model.offset
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess

    def _separate(self, X_mag_pad, roi_size):
        X_dataset = []
        patches = (X_mag_pad.shape[2] - 2 * self.offset) // roi_size
        for i in range(patches):
            start = i * roi_size
            X_mag_crop = X_mag_pad[:, :, start:start + self.cropsize]
            X_dataset.append(X_mag_crop)

        X_dataset = np.asarray(X_dataset)

        self.model.eval()
        with torch.no_grad():
            mask = []
            # To reduce the overhead, dataloader is not used.
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(X_batch).to(self.device)

                pred = self.model.predict_mask(X_batch)

                pred = pred.detach().cpu().numpy()
                pred = np.concatenate(pred, axis=2)
                mask.append(pred)

            mask = np.concatenate(mask, axis=2)

        return mask

    def _preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def _postprocess(self, mask, X_mag, X_phase):
        if self.postprocess:
            mask = spec_utils.merge_artifacts(mask)

        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)

        return y_spec, v_spec

    def separate(self, X_spec):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask = self._separate(X_mag_pad, roi_size)
        mask = mask[:, :, :n_frame]

        y_spec, v_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec

    def separate_tta(self, X_spec):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask = self._separate(X_mag_pad, roi_size)

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask_tta = self._separate(X_mag_pad, roi_size)
        mask_tta = mask_tta[:, :, roi_size // 2:]
        mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5

        y_spec, v_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print('loading model...', end=' ')
        device = torch.device('cuda:0')
        self.n_fft = 2048
        model_dir = "models/baseline.pth"
        gpu = 1
        batchsize = 4
        cropsize = 256
        postprocess = False

        model = nets.CascadedNet(self.n_fft, 32, 128)
        model.load_state_dict(torch.load(model_dir, map_location=device))
        if gpu >= 0:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                model.to(device)
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device('mps')
                model.to(device)
        print('done')

        self.sp = Separator(model, device, batchsize, cropsize, postprocess)

    def predict(
            self,
            audio_file : Path = Input(
            description="An audio file that will separated",
            default=None),
            result : str = Input(
            description="What result file you want",
            choices=["all", "instrument", "vocal"],
            default="all"
            )
    ) -> List[Path]:
        tta = True
        sr = 44100
        hop_length = 1024
        output_dir="result"
        unique_id = uuid.uuid4().hex
        # infernya
        print('loading wave source...', end=' ')
        X, sr = librosa.load(
            audio_file, sr=sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
        # basename = os.path.splitext(os.path.basename(input))[0]
        print('done')

        if X.ndim == 1:
            # mono to stereo
            X = np.asarray([X, X])

        print('stft of wave source...', end=' ')
        X_spec = spec_utils.wave_to_spectrogram(X, hop_length, self.n_fft)
        print('done')

        sp = self.sp

        if tta:
            y_spec, v_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec = sp.separate(X_spec)

        print('validating output directory...', end=' ')
        if output_dir != "":  # modifies output_dir if theres an arg specified
            output_dir = output_dir.rstrip('/') + '/'
            os.makedirs(output_dir, exist_ok=True)

        if (result == "instrument" or result == "all"):
            print('inverse stft of instruments...', end=' ')
            instrument_path = '{}{}_Instruments.wav'.format(output_dir, unique_id)
            print(f'ini_path {instrument_path}')
            wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=hop_length)
            sf.write(instrument_path, wave.T, sr)
    
        if (result == "vocal" or result == "all"):
            print('inverse stft of vocals...', end=' ')
            vocals_path = '{}{}_Vocals.wav'.format(output_dir, unique_id)
            wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=hop_length)
            sf.write(vocals_path, wave.T, sr)

        # index
        # [0] = audio instrumen
        # [1] = audio vocal
        if result == "all":
            output = [Path(instrument_path), Path(vocals_path)]
        elif result == "instrument":
            output = [Path(instrument_path)]
        else:
            output = [Path(vocals_path)]
        return output