# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 13:12:12 2025

@author: alberto
"""

import torch
import librosa as li
import soundfile as sf
import argparse

parser = argparse.ArgumentParser(description='Example python script to generate & manipulate audio with RAVE.')
parser.add_argument('--model', type=str, help='The model file (needs non-streaming model?)')
parser.add_argument('--input', type=str, help='Input wav (please pre-convert with ffmpeg)')
parser.add_argument('--output', type=str, default="<<unset>>", help='Default = [input]_out.wav')
parser.add_argument('--duration', type=float, default=9000001.0, help='Cap input wav length')
args = parser.parse_args()

if not (args.model and args.input):
#    parser.print_help()
    parser.error('Needs more arguments.')
if args.output == "<<unset>>":
    args.output = args.input+"_out.wav"

torch.set_grad_enabled(False)
model = torch.jit.load(args.model).eval()

way = 2

if way == 1:
        x = li.load(args.input, mono=(True if model.n_channels == 1 else False), sr=44100, duration=args.duration)[0]
        x = torch.from_numpy(x).reshape(1,model.n_channels,-1)
        z = model.encode(x)
        
        ## skews latent dimension #6 from -2 to 2 - uncomment if desired
        # z[:, 5] += torch.linspace(-2,2,z.shape[-1])
        
        ## interlaces data with zeros
        # interlaced_z = torch.zeros(z.shape[0], z.shape[1], z.shape[2]*2, device=z.device, dtype=z.dtype)
        # interlaced_z[:, :, ::2] = z
        # z = interlaced_z
        
        ## elongate 2x
        # z = z.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(z.shape[0], z.shape[1], -1)
        
        ## skews latent dimension #4 by +2 and interlaces it
        # zz = z.clone()
        # zz[:, 3] += torch.linspace(2,2,z.shape[-1])
        # z[:, :, 1::2] = zz[:, :, 1::2]
        
        y = model.decode(z).detach().numpy().reshape(-1, model.n_channels)
        sf.write(args.output, y, 44100)
        

if way == 2:
        x, sr = li.load(args.input,sr=44100)
        x = torch.from_numpy(x).reshape(1,1,-1)
        
        # encode and decode the audio with RAVE
        z = model.encode(x)
        x_hat = model.decode(z).numpy().reshape(-1)
        
        sf.write(args.output, x_hat, 44100)

