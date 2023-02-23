#coding: utf-8

import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
#from colossalai.utils import get_dataloader
import librosa
from g2p_en import G2p

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from text_utils import TextCleaner
np.random.seed(1)
random.seed(1)
DEFAULT_DICT_PATH = osp.join(osp.dirname(__file__), 'word_index_dict.txt')
SPECT_PARAMS = {
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 256
}
MEL_PARAMS = {
    "n_mels": 100,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 256
}


class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 dict_path=DEFAULT_DICT_PATH,
                 sr=24000
                ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner(dict_path)
        self.sr = sr

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
        self.mean, self.std = -4, 4
        
        self.g2p = G2p()
        self.MAX_WAV_VALUE = 32768.0

    


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        def melSpectrogram(y, n_fft=1024, num_mels=100, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000, center=False):
            mel_basis = {}
            hann_window = {}
            def dynamic_range_compression(x, C=1, clip_val=1e-5):
                return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


            def dynamic_range_decompression(x, C=1):
                return np.exp(x) / C


            def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
                return torch.log(torch.clamp(x, min=clip_val) * C)


            def dynamic_range_decompression_torch(x, C=1):
                return torch.exp(x) / C
            
            def spectral_normalize_torch(magnitudes):
                output = dynamic_range_compression_torch(magnitudes)
                return output


            def spectral_de_normalize_torch(magnitudes):
                output = dynamic_range_decompression_torch(magnitudes)
                return output
            if torch.min(y) < -1.:
                print('min value is ', torch.min(y))
            if torch.max(y) > 1.:
                print('max value is ', torch.max(y))

            global mel_basis, hann_window
            if fmax not in mel_basis:
                mel = librosa.fiters.mel(sampling_rate, n_fft, num_mels, fmin, fmax)
                mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
                hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

            y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
            y = y.squeeze(1)

            # complex tensor as default, then use view_as_real for future pytorch compatibility
            spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                            center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
            spec = torch.view_as_real(spec)
            spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

            spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
            spec = spectral_normalize_torch(spec)

            return spec
        
        
        data = self.data_list[idx]
        wave, text_tensor, speaker_id = self._load_tensor(data)
        wave = wave / self.MAX_WAV_VALUE

        
        wave_tensor = torch.FloatTensor(wave)
        mel_tensor = melSpectrogram(wave_tensor)

        if (text_tensor.size(0)+1) >= (mel_tensor.size(1) // 3):
            mel_tensor = F.interpolate(
                mel_tensor.unsqueeze(0), size=(text_tensor.size(0)+1)*3, align_corners=False,
                mode='linear').squeeze(0)

        acoustic_feature = (torch.log(1e-5 + mel_tensor) - self.mean)/self.std

        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]

        return wave_tensor, acoustic_feature, text_tensor, data[0]

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(wave_path)

        # phonemize the text
        ps = self.g2p(text.replace('-', ' '))
        if "'" in ps:
            ps.remove("'")
        text = self.text_cleaner(ps)
        blank_index = self.text_cleaner.word_index_dictionary[" "]
        text.insert(0, blank_index) # add a blank at the beginning (silence)
        text.append(blank_index) # add a blank at the end (silence)
        
        text = torch.LongTensor(text)

        return wave, text, speaker_id




class Collater(object):
    """
    Args:
      return_wave (bool): if true, will return the wave data along with spectrogram. 
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave

    def __call__(self, batch):
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        paths = ['' for _ in range(batch_size)]
        for bid, (_, mel, text, path) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            assert(text_size < (mel_size//2))

        if self.return_wave:
            waves = [b[0] for b in batch]
            return texts, input_lengths, mels, output_lengths, paths, waves

        return texts, input_lengths, mels, output_lengths



def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = MelDataset(path_list, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
