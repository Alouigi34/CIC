# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:58:35 2024

@author: alberto
"""

_path_ = r"C:\Users\dynal\OneDrive\Desktop\v10_characteristics_from_encoder - Copy\test_new3"

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.stats import skew, kurtosis
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import torch
import soundfile as sf
import numpy as np
import IPython.display as ipd  # For audio playback in notebooks
import matplotlib.pyplot as plt
import time


#########################################   If using mfcc:
encoding_method="mfcc"
mfcc_characteristics = 110
new_sampling_rate = 41000

#########################################   If using encodec:
#encoding_method="encodec"
encodec_bandwidth = 6.0 # For example, a bandwidth of 24 corresponds to 32 characteristics
#############################################################

if encoding_method=="mfcc":
    past_information_samples = 600   #No of samples used as knowledge
    future_information_samples = 180   #No of samples to predict in the future
    epochs_number = 10
else:
    past_information_samples = 600   #No of samples used as knowledge
    future_information_samples = 60   #No of samples to predict in the future   
    epochs_number = 20
#############################################################

train_using_LSTM = False
train_using_TCN = False
train_using_advanced_TCN = False
train_using_AdvancedTCN_faster = True
train_using_transformer = False
train_using_transformer2 = False
#############################################################

z_score_normalization = True
min_max_normalization = False
#############################################################

retrain=False
#############################################################

# File paths
audio_file = _path_ + r'\song1.wav'
output_file = _path_ + r'\predicted_song1_portion.wav'
output_file2 = _path_ + r'\direct_from_mfcc_song1_portion.wav'


'''
END USER PARAMETERS
'''

## Create models subfolder  1st AI
import os 
# checking if the directory exist or not. 
if not os.path.exists(_path_ +r"\AImodel1"):      
    # if the directory is not present  then create it. 
    os.makedirs(_path_ +r"\AImodel1") 
    


import glob

# Directory containing all songs
songs_dir = os.path.join(_path_, 'training_songs')

# Get a list of all .wav files in the songs directory
audio_files = glob.glob(os.path.join(songs_dir, '*.wav'))

# Initialize a list to hold features from all songs
all_notes_list = []

for audio_file in audio_files:
    print(f"Processing {audio_file}...")
    
    if encoding_method == "encodec":
        # Instantiate a pretrained EnCodec model
        model_enc = EncodecModel.encodec_model_24khz()
        model_enc.set_target_bandwidth(encodec_bandwidth)  # Set target bandwidth to 6 kbps

        # Load and pre-process the audio waveform
        wav, sr = torchaudio.load(audio_file)
        wav = convert_audio(wav, sr, model_enc.sample_rate, model_enc.channels)
        wav = wav.unsqueeze(0)  # Add batch dimension

        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = model_enc.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

        # Convert the codes to a numpy array
        codes_numpy = codes.squeeze(0).cpu().numpy()  # Remove batch dimension
        print(f"Shape of the encoded matrix: {codes_numpy.shape}")
        
        # Optional: Visualize (consider limiting or skipping for multiple songs)
        # plt.imshow(codes_numpy, aspect='auto', cmap='viridis')
        # plt.title("Visualized Discrete Codes from EnCodec")
        # plt.xlabel("Time Steps")
        # plt.ylabel("Codebooks")
        # plt.colorbar()
        # plt.show()
        
        all_notes = codes_numpy.T.copy()  # Transpose to shape (Time Steps, Codebooks)

    elif encoding_method == "mfcc":
        # Define MFCC to Mel spectrogram and waveform conversion functions if not already defined
        def mfcc_to_mel_spectrogram(mfcc):
            mel_spec = librosa.feature.inverse.mfcc_to_mel(mfcc.T)
            return mel_spec

        def mel_to_waveform(mel_spec, sr):
            return librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr, n_iter=280)
    
        # Load and preprocess audio
        y, sr = librosa.load(audio_file)
        y2 = librosa.resample(y, orig_sr=sr, target_sr=new_sampling_rate)
        mfcc = librosa.feature.mfcc(y=y2, sr=new_sampling_rate, n_mfcc=mfcc_characteristics)
        all_notes = mfcc.T.copy()  # Transpose to shape (Time Steps, MFCC Features)

    '''
    Enlarge the signal, (maybe to train better?)
    '''
    # Number of duplications
    num_duplicates = 1
    # Data Augmentation: Duplicate if necessary
    all_notes = np.tile(all_notes, (num_duplicates, 1))  # Adjust duplication as needed
    all_notes = all_notes.astype(float)
    all_notes_list.append(all_notes)

# Concatenate all notes from all songs
all_notes_combined = np.concatenate(all_notes_list, axis=0)
print(f"Combined all_notes shape: {all_notes_combined.shape}")


all_notes = all_notes_combined.copy()

# Convert all_notes to float before normalization
all_notes = all_notes.astype(float)
print(all_notes.shape)
initial_song_data = all_notes.copy()


'''
THIS IS TO TEST PREDICTION ACCURACY FOR A MORE SIMPLE USE CASE with 3 periodical characteristics
'''
# #################################################################################################
# # Number of samples
# num_samples = 10000

# # Generate a time vector
# t = np.linspace(0, 400 * np.pi, num_samples)  # 0 to 4π

# # Generate two sine waves with different frequencies and phases
# all_notes= np.zeros((num_samples, 3))
# all_notes[:, 0] = 1000.0*np.sin(t)  # Sine wave for the first column
# all_notes[:, 1] = 0.5*all_notes[:, 0] +500.0*np.sin(2.5 * t + np.pi / 3)  # Sine wave with double frequency and phase shift for the second column
# all_notes[:, 2] = 0.5*all_notes[:, 1]+ 400.0*np.sin(8 * t + np.pi / 4)  # Sine wave with double frequency and phase shift for the second column
# #################################################################################################


if min_max_normalization == True:
    # Normalize features and save to CSV
    def normalize(arr, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = np.max(arr) - np.min(arr)
        characteristic_max = np.max(arr)
        characteristic_min = np.min(arr)
        if np.isnan(diff_arr) == False:
            for i in arr:
                temp = (((i - np.min(arr)) * diff) / diff_arr) + t_min
                norm_arr.append(temp)
        if np.isnan(diff_arr) == True:
            for i in arr:
                norm_arr.append(0.0)
        return norm_arr, characteristic_max, characteristic_min
    
    characteristics_max = []
    characteristics_min = []
    
    for i in range(len(all_notes[0])):
        range_to_normalize = (0, 1)
        all_notes[:, i], characteristic_max, characteristic_min = normalize(all_notes[:, i],
                                                                            range_to_normalize[0],
                                                                            range_to_normalize[1])
        characteristics_max.append(characteristic_max)
        characteristics_min.append(characteristic_min)
    
    all_notes = np.array(all_notes)
    all_notes[np.isnan(all_notes)] = 0
    
    



if z_score_normalization == True:
    # Z-score normalization function
    def z_score_normalize(arr):
        mean = np.mean(arr)
        std_dev = np.std(arr)
        normalized_arr = (arr - mean) / std_dev
        return normalized_arr, mean, std_dev
    
    # Robust scaling normalization function
    def robust_scale_normalize(arr):
        median = np.median(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        normalized_arr = (arr - median) / iqr
        return normalized_arr, median, iqr
    
    # Updated main code for choosing normalization method
    characteristics_params = []  # Store mean, std_dev or median, IQR based on method
    
    # Choose normalization method
    normalization_method = "z_score"  # Choose either "z_score" or "robust_scale"
    
    for i in range(len(all_notes[0])):
        if normalization_method == "z_score":
            all_notes[:, i], param1, param2 = z_score_normalize(all_notes[:, i])
        elif normalization_method == "robust_scale":
            all_notes[:, i], param1, param2 = robust_scale_normalize(all_notes[:, i])
        
        characteristics_params.append((param1, param2))  # Store parameters for denormalization
    
    # Replace NaNs with 0 in the final normalized data
    all_notes = np.array(all_notes)
    all_notes[np.isnan(all_notes)] = 0
    
    


# Save to CSV
np.savetxt(_path_+r'\summary_notes_audio.csv', all_notes, delimiter=',')

# Load and print the normalized data to verify
datax = np.loadtxt(_path_+r'\summary_notes_audio.csv', delimiter=',')
#datax = datax.astype(int)



print(datax.shape)
number_of_characteristics = int(len(all_notes[0]))




"""
#########################################################################################################
#########################################################################################################
Step A: Training phase
-> step A2 : Train AI model 1 
#########################################################################################################
#########################################################################################################
"""
import gc
import numpy as np
import tensorflow as tf
print(tf. __version__)
import keras
import time
#tf.compat.v1.disable_eager_execution()

#Before do anything else do not forget to reset the backend for the next iteration (rerun the model)
keras.backend.clear_session()#####################################

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable(package="MyLayers")
class MultidimensionalLSTM(tf.keras.Model):
    def __init__(self, hidden_size, hidden_size2, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        #self.lstm = tf.keras.layers.LSTM(hidden_size,activation='silu', return_sequences=True)        
        self.lstm = tf.keras.layers.LSTM(hidden_size,activation='relu', return_sequences=True)        
        self.dense = tf.keras.layers.Dense(output_size,activation='linear')
        self.lstm2 = tf.keras.layers.LSTM(hidden_size, activation='sigmoid', return_sequences=True)
    def call(self, inputs):
        x = self.lstm(inputs) 
        #x = self.lstm2(x)
        #x = self.linear(x)   
        x = self.dense(x[:, -1, :])
        x = tf.reshape(x, (-1, future_information_samples,1))  # Reshape to (batch_size, 2, output_size)
        return x
    
from tensorflow.keras.layers import BatchNormalization    
@keras.saving.register_keras_serializable(package="MyLayers")
class TCN(tf.keras.Model):
    def __init__(self, num_channels, kernel_size, output_size):
        super(TCN, self).__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.output_size = output_size

        self.conv1d_layers = []
        if isinstance(num_channels, int):  # Handling single integer case
            num_channels = [num_channels]
        for i, channels in enumerate(num_channels):
            self.conv1d_layers.append(
                tf.keras.layers.Conv1D(channels, kernel_size, padding='causal', activation='relu', dilation_rate=2**i)
            )
            self.conv1d_layers.append(BatchNormalization())
        self.dense = tf.keras.layers.Dense(output_size, activation='linear')

    def call(self, inputs):
        x = inputs
        for layer in self.conv1d_layers:
            x = layer(x)
        x = self.dense(x[:, -1, :])
        return x


@keras.saving.register_keras_serializable(package="MyLayers")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Add & Norm
        
        ffn_output = self.ffn(out1)  # Feed Forward Network
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Add & Norm

@keras.saving.register_keras_serializable(package="MyLayers")
class TransformerModel(tf.keras.Model):
    def __init__(self, num_heads, ff_dim, output_size, num_transformer_blocks, embed_dim, rate=0.1):
        super(TransformerModel, self).__init__()

        # Positional Embedding to capture temporal structure
        self.positional_encoding = tf.keras.layers.Embedding(input_dim=10000, output_dim=embed_dim)

        # Stacking transformer blocks
        self.transformer_blocks = [TransformerBlock(embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    ff_dim=ff_dim, rate=rate)
                                   for _ in range(num_transformer_blocks)]

        # Output layer for sequence prediction
        self.dense = tf.keras.layers.Dense(output_size, activation='linear')

    def call(self, inputs, training=False):
        # Create a position tensor (0, 1, 2, ..., seq_len)
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        
        # Add positional encoding to inputs
        x = inputs + self.positional_encoding(positions)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training)

        x = self.dense(x[:, -1, :])  # Use the last time step's output
        return x


# Sample Usage:
# transformer_model = TransformerModel(num_heads=4, ff_dim=128, output_size=future_information_samples, num_transformer_blocks=2, embed_dim=128)



import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel2(nn.Module):
    def __init__(self, num_heads, ff_dim, output_size, num_transformer_blocks, embed_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(embed_dim, embed_dim)  # Assuming inputs are already features (no word embeddings)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_transformer_blocks)
        self.fc_out = nn.Linear(embed_dim, output_size)

    def forward(self, src):
        # src shape: (seq_len, batch_size, embed_dim)
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output[-1])  # Use the last time step's output
        return output




@keras.saving.register_keras_serializable(package="MyLayers")
class AdvancedTCN(tf.keras.Model):
    def __init__(self, num_channels, kernel_size, output_size, dilations):
        super(AdvancedTCN, self).__init__()

        self.conv1d_layers = []
        for i, channels in enumerate(num_channels):
            dilation_rate = dilations[i] if i < len(dilations) else 1
            self.conv1d_layers.append(
                tf.keras.layers.Conv1D(channels, kernel_size, padding='causal', activation='relu', dilation_rate=dilation_rate)
            )
            self.conv1d_layers.append(tf.keras.layers.BatchNormalization())

        # Dense layer for output
        self.dense = tf.keras.layers.Dense(output_size, activation='linear')

        # Add residual connections
        self.residual_connection = tf.keras.layers.Conv1D(num_channels[-1], kernel_size=1, padding='same', activation=None)

    def call(self, inputs):
        x = inputs
        for conv1d_layer in self.conv1d_layers:
            x = conv1d_layer(x)

        # Apply residual connection
        residual_output = self.residual_connection(inputs)
        x += residual_output

        x = self.dense(x[:, -1, :])
        return x

# Sample Usage:
# tcn_model = AdvancedTCN(num_channels=[64, 128, 256], kernel_size=3, output_size=future_information_samples, dilations=[1, 2, 4])


import torch
import torch.nn as nn

class AdvancedTCN_faster(nn.Module):
    def __init__(self, num_channels, kernel_size, output_size, dilations, input_channels):
        super(AdvancedTCN_faster, self).__init__()
        
        layers = []
        for i, channels in enumerate(num_channels):
            dilation_rate = dilations[i] if i < len(dilations) else 1
            layers.append(
                nn.Conv1d(in_channels=num_channels[i-1] if i > 0 else input_channels,
                          out_channels=channels, 
                          kernel_size=kernel_size, 
                          padding=(kernel_size - 1) * dilation_rate,  # 'causal' padding equivalent
                          dilation=dilation_rate)
            )
            layers.append(nn.BatchNorm1d(channels))
            layers.append(nn.ReLU())

        self.conv1d_layers = nn.Sequential(*layers)

        # Dense layer for output
        self.dense = nn.Linear(num_channels[-1], output_size)

        # Residual connection
        self.residual_connection = nn.Conv1d(in_channels=input_channels, out_channels=num_channels[-1], kernel_size=1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        x = x.transpose(1, 2)  # PyTorch expects (batch_size, num_features, sequence_length)
        residual = self.residual_connection(x)

        out = self.conv1d_layers(x)

        # Adjust residual to match out sequence length (cropping)
        if residual.size(2) != out.size(2):
            min_length = min(residual.size(2), out.size(2))
            residual = residual[:, :, :min_length]
            out = out[:, :, :min_length]

        # Residual connection addition
        out = out + residual

        # Flatten the final layer and apply dense
        out = out[:, :, -1]  # Take the last time step (sequence_length dimension)
        out = self.dense(out)
        
        return out










######################################################################
##############  INPUT DATA
######################################################################
#This is the time buffer (how many time samples will be used):
buffer_length=past_information_samples
#This is the ammount of the characteristics choosen starting from the all_characteristics index:
characteristics_buff=number_of_characteristics
#This is the len of the characteristics: 
#(choose number_of_characteristics if duration should be also considered)
#(choose 88 if duration shouldn't be considered)
#(choose 41 to include characteristics between 41-characteristics_buff:41
all_characteristics=number_of_characteristics
inputs=np.full((len(all_notes)-buffer_length-future_information_samples,buffer_length,characteristics_buff), 0.0)

# Separate to inputs-outputs
for time_index in range(len(all_notes)-buffer_length-future_information_samples): #number of changes
    input_temp=all_notes[time_index:time_index+buffer_length,all_characteristics-characteristics_buff:all_characteristics]
    inputs[time_index,0:buffer_length,0:characteristics_buff]=(input_temp)

# len(all_notes) samples, each with buffer_length time steps and number_of_characteristics features
input_data1 = inputs

################################################################################
################################################################################ 

######################################################################
##############  OUTPUT DATA
######################################################################

output_data1=[]
num_samples = len(all_notes) - buffer_length - future_information_samples  # Adjust for the two future values
for i in range(number_of_characteristics):
    #This is the time buffer (how many time samples will be used):
    buffer_length=past_information_samples
    outputs=np.full((len(all_notes)-buffer_length,1,1), 0.0)
    
    SPESIFIC_CHARACTERISTIC_TO_PREDICT = i
    
    outputs = np.full((num_samples, future_information_samples, 1), 0.0)
    for time_index in range(num_samples):
        output_temp = all_notes[time_index + buffer_length:time_index + buffer_length + future_information_samples, i]
        for x in range(future_information_samples):
            outputs[time_index, x, 0] = output_temp[x]
    output_data1.append(outputs)
################################################################################
################################################################################   

################################################################################   
################################################################################
################################################################################   

  
if retrain == True:
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from torch.utils.data import DataLoader, TensorDataset
    import gc

    # Loop through characteristics to train the models
    for i in range(0, number_of_characteristics):
        # Handle TensorFlow/Keras models
        if train_using_LSTM == True:
            model = MultidimensionalLSTM(hidden_size=50, hidden_size2=32, output_size=future_information_samples)
        if train_using_TCN == True:
            model = TCN(num_channels=20, kernel_size=58, output_size=future_information_samples)
        if train_using_transformer == True:
            embed_dim = 90  # Make sure this matches the number of features
            model = TransformerModel(num_heads=2, ff_dim=32, output_size=future_information_samples, num_transformer_blocks=2, embed_dim=embed_dim)
        if train_using_transformer2 == True:
            # Transformer model variant
            embed_dim = 90  # Number of features or characteristics
            output_size = future_information_samples  # Size of the output prediction
            num_heads = 4  # Number of heads for multi-head attention
            ff_dim = 128  # Feedforward network dimension
            num_transformer_blocks = 2  # Number of Transformer blocks
            model = TransformerModel(num_heads=num_heads, ff_dim=ff_dim, output_size=output_size, num_transformer_blocks=num_transformer_blocks, embed_dim=embed_dim)
        if train_using_advanced_TCN == True:
            model = AdvancedTCN(num_channels=[64, 128, 256], kernel_size=3, output_size=future_information_samples, dilations=[1, 2, 4])

        # TensorFlow/Keras models training
        if any([train_using_LSTM, train_using_TCN, train_using_transformer, train_using_transformer2, train_using_advanced_TCN]):
            model.compile(optimizer=tf.keras.optimizers.Adam(0.0002), loss=tf.keras.losses.MeanSquaredError(), metrics=['mean_squared_error'])
            
            print(f"################ Characteristic: {i} ########################")
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            checkpoint = ModelCheckpoint(f'best_model_{i}.h5', monitor='val_loss', save_best_only=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            
            # Train the TensorFlow model
            model.fit(input_data1, output_data1[i], epochs=epochs_number, callbacks=[early_stopping, checkpoint, reduce_lr])
            
            # Save the TensorFlow model
            path = _path_ + r"\AImodel1\\"
            model.save(path + f"custom_model_{i}.keras")

            # Clean up memory
            del model
            tf.keras.backend.clear_session()
            gc.collect()

        # Handle PyTorch model (AdvancedTCN_faster) training
        elif train_using_AdvancedTCN_faster == True:
            # Create the model and move it to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AdvancedTCN_faster(num_channels=[128, 512, 1024], kernel_size=5, output_size=future_information_samples, dilations=[1, 4, 8],input_channels = len(all_notes[0])).to(device)
            #model = AdvancedTCN_faster(num_channels=[64, 128, 256], kernel_size=3, output_size=future_information_samples, dilations=[1, 2, 4], input_channels=len(all_notes[0])).to(device)
            print(f"################ Characteristic: {i} ########################")


            if i == 0:
                # Convert input data and output data to PyTorch tensors
                input_tensor = torch.tensor(input_data1, dtype=torch.float32).to(device)
            output_tensor = torch.tensor(output_data1[i], dtype=torch.float32).to(device)

            # Create a TensorDataset and DataLoader
            dataset = TensorDataset(input_tensor, output_tensor)
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Define optimizer and loss for PyTorch models
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
            criterion = nn.MSELoss()

            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # Training loop for PyTorch model
            for epoch in range(epochs_number):  # Number of epochs
                model.train()
                train_loss = 0.0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
            
                    optimizer.zero_grad()  # Reset gradients
                    outputs = model(inputs)  # Forward pass
                    
                    # Reshape outputs to match the target shape
                    outputs = outputs.unsqueeze(-1)  # Add a new dimension to make the shape [batch_size, sequence_length, 1]
                    
                    loss = criterion(outputs, targets)  # Compute loss
                    loss.backward()  # Backpropagate
                    optimizer.step()  # Update model weights
            
                    train_loss += loss.item()
            
                # Optionally, log the training loss per epoch
                print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}")


            torch.cuda.empty_cache()
            # Save PyTorch model
            path = _path_ + r"\AImodel1\\"
            torch.save(model.state_dict(), path + f"custom_model_{i}.pth")

            # Clear memory for PyTorch
            del model
            # After training for characteristic i, release the memory
            del output_tensor
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(5)


    # Load TensorFlow and PyTorch models
    loaded_models = []
    for i in range(0, number_of_characteristics):
        if any([train_using_LSTM, train_using_TCN, train_using_transformer, train_using_transformer2, train_using_advanced_TCN]):
            # Load TensorFlow model
            path = _path_ + r"\AImodel1\\"
            loaded_models.append(tf.keras.models.load_model(path + f"custom_model_{i}.keras"))
        elif train_using_AdvancedTCN_faster == True:
            # Load PyTorch model
            # model = AdvancedTCN_faster(num_channels=[64, 128, 256], kernel_size=3, output_size=future_information_samples, dilations=[1, 2, 4],input_channels = len(all_notes[0]))
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AdvancedTCN_faster(num_channels=[128, 512, 1024], kernel_size=5, output_size=future_information_samples, dilations=[1, 4, 8],input_channels = len(all_notes[0]))
            path = _path_ + r"\AImodel1\\"
            model.load_state_dict(torch.load(path + f"custom_model_{i}.pth", map_location=device))
            model = model.to(device)
            loaded_models.append(model)

    
  
    
  
    
    #Check 
    # print("if following are equal means that loaded models are the same with the saved")
    # test_data = input_data[184,:,:]
    # test_data = test_data.reshape((1, buffer_length, characteristics_buff))   
    # print(model[1].predict(test_data, verbose=1))
    # print(loaded_models[1].predict(test_data, verbose=1))


if retrain == False:
    loaded_models = []
    for i in range(0, number_of_characteristics):
        if any([train_using_LSTM, train_using_TCN, train_using_transformer, train_using_transformer2, train_using_advanced_TCN]):
            # Load TensorFlow model
            path = _path_ + r"\AImodel1\\"
            loaded_models.append(tf.keras.models.load_model(path + f"custom_model_{i}.keras"))
        elif train_using_AdvancedTCN_faster == True:
            # Load PyTorch model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # model = AdvancedTCN_faster(num_channels=[64, 128, 256], kernel_size=3, output_size=future_information_samples, dilations=[1, 2, 4],input_channels = len(all_notes[0]))
            model = AdvancedTCN_faster(num_channels=[128, 512, 1024], kernel_size=5, output_size=future_information_samples, dilations=[1, 4, 8],input_channels = len(all_notes[0]))
            path = _path_ + r"\AImodel1\\"
            model.load_state_dict(torch.load(path + f"custom_model_{i}.pth", map_location=device))
            model = model.to(device)
            loaded_models.append(model)

    

"To check if the predictions are same with the initial song"
check_prediction_accuracy = False
if check_prediction_accuracy == True:
    t0 = time.time()
    final_predicted = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for index in range(0, 20):  # For each state change
        predictNextSequence = []
        real = []
        test_data = input_data1[index, :, :]
        test_data = test_data.reshape((1, buffer_length, characteristics_buff))
        
        for i in range(0, number_of_characteristics):  # For each characteristic
            if train_using_AdvancedTCN_faster:
                # PyTorch prediction for AdvancedTCN_faster model
                test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
                model = loaded_models[i].to(device)
                model.eval()  # Set the model to evaluation mode
                
                with torch.no_grad():
                    output = model(test_tensor)
                
                # Convert the PyTorch output back to numpy
                output = output.cpu().numpy()
                temp = np.squeeze(output)
                temp[temp < 0.01] = 0
                predictNextSequence.append(temp)
                
            else:
                # TensorFlow prediction for other models
                temp = loaded_models[i].predict(test_data, verbose=1)
                temp[temp < 0.01] = 0
                predictNextSequence.append(temp)
            
            real.append(output_data1[i][index, :, :])
        
        # Format the predictions
        final_predicted.append(np.transpose(predictNextSequence, axes=(2, 1, 0)))

        print(f"Prediction completed for index: {index}")

    t1 = time.time()
    print(f'Prediction duration = {t1 - t0}')

    # Restore final_predicted to have the same format as all_notes
    final_predicted_numpy = np.array(final_predicted)
    final_predicted_numpy = final_predicted_numpy[:, 0, 0, :]

    # Denormalize in order to reconstruct the piece
    ####################### Data De-Normalization ################################
    if min_max_normalization == True:
        # Explicit function to denormalize array
        def denormalize(arr, t_min, t_max):
            denorm_arr = []
            diff = 1
            diff_arr = t_max - t_min
            if not np.isnan(diff_arr):
                for i in arr:
                    temp = t_min + ((i - 0) * diff_arr / diff)
                    denorm_arr.append(temp)
            return denorm_arr

        final_predicted_numpy_actual = final_predicted_numpy.copy()
        # De-Normalize each feature independently:
        for i in range(number_of_characteristics):
            final_predicted_numpy_actual[:, i] = denormalize(
                final_predicted_numpy[:, i], characteristics_min[i], characteristics_max[i]
            )

        final_predicted_numpy_actual = np.array(final_predicted_numpy_actual)
        final_predicted_numpy_actual[np.isnan(final_predicted_numpy_actual)] = 0

    if z_score_normalization == True:
        # Z-score denormalization
        def z_score_denormalize(arr, mean, std_dev):
            return arr * std_dev + mean

        # Update to denormalize based on the chosen method
        final_predicted_numpy_actual = final_predicted_numpy.copy()

        # Loop over the characteristics to denormalize each one
        for i in range(number_of_characteristics):
            value = final_predicted_numpy_actual[:, i]
            param1, param2 = characteristics_params[i]  # Retrieve the saved normalization parameters
            final_predicted_numpy_actual[:, i] = z_score_denormalize(value, param1, param2)

        final_predicted_numpy_actual = np.array(final_predicted_numpy_actual)
        final_predicted_numpy_actual[np.isnan(final_predicted_numpy_actual)] = 0

    # Export predicted notes to CSV file
    print("Exporting to CSV. Note that the exported file has buffer_length fewer elements than the original")
    np.savetxt('final_predicted_AI_from_training.csv', final_predicted_numpy_actual)



"""
#########################################################################################################
#########################################################################################################
Step A5 : Convert TensorFlow models to TensorFlow Lite models for faster predictions (if applicable)
#########################################################################################################
#########################################################################################################
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer


class LiteModel:
    
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))
    
    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] 
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))
    
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]
        
    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out_shape = (count,) + tuple(self.output_shape[1:])
        out = np.zeros(out_shape, dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i+1])
            self.interpreter.invoke()
            output_tensor = self.interpreter.get_tensor(self.output_index)
            out[i] = output_tensor.reshape(self.output_shape[1:])
        return out  
    
    
    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]


### Convert TensorFlow models to TensorFlow Lite models
lmodels = []
if train_using_LSTM or train_using_TCN or train_using_transformer or train_using_transformer2 or train_using_advanced_TCN:
    for i in range(number_of_characteristics):
        print(f"Converting model {i} to TensorFlow Lite.")
        lmodels.append(LiteModel.from_keras_model(loaded_models[i]))


"""
#########################################################################################################
#########################################################################################################
Step B: Prediction using TensorFlow Lite models and PyTorch models (if applicable)
#########################################################################################################
#########################################################################################################
"""

# example 1 is predictions on random user choices
# example 3 is to check model's accuracy with a given piece
example = 3

if example == 3:
    ################################################   example input 3 (training song's pattern)
    # Initialize:
    user_input_time_step = []
    for i in range(buffer_length):
        user_input_time_step.append(np.zeros(number_of_characteristics))
    
    # Add some random notes (with hit power). 
    index_check = 4800
    user_input_time_step =  input_data1[index_check, :, :]

# Normalize user input:
normalized_user_input_time_step = np.array(user_input_time_step.copy())

##### Predict for TensorFlow Lite models
if True:
    normalized_user_input_time_step_reshaped = normalized_user_input_time_step.reshape((1, buffer_length, characteristics_buff))    
    predictNextSequence = []

    # If using TensorFlow Lite models
    if train_using_LSTM or train_using_TCN or train_using_transformer or train_using_transformer2 or train_using_advanced_TCN:
        for i in range(0, number_of_characteristics):  # For each characteristic
            temp = lmodels[i].predict(normalized_user_input_time_step_reshaped)
            predictNextSequence.append(temp)

    # Handle PyTorch models
    elif train_using_AdvancedTCN_faster:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_tensor = torch.tensor(normalized_user_input_time_step_reshaped, dtype=torch.float32).to(device)
        predictNextSequence = []

        for i in range(number_of_characteristics):
            model = loaded_models[i].to(device)
            model.eval()  # Set the model to evaluation mode

            with torch.no_grad():
                output = model(test_tensor)
                output = output.cpu().numpy()
                predictNextSequence.append(np.squeeze(output))



# Handle denormalization if needed
####################### Data De-Normalization ################################
if min_max_normalization:
    def denormalize(arr, t_min, t_max):
        diff_arr = t_max - t_min
        if not np.isnan(diff_arr):
            return t_min + ((arr - 0) * diff_arr)
        return arr
    
    final_predicted_numpy_actual = predictNextSequence.copy()
    
    # De-normalize each feature independently:
    for i in range(number_of_characteristics):
        value = (final_predicted_numpy_actual[i])
        if example == 3:
            final_predicted_numpy_actual[i] = denormalize(value, characteristics_min[i], characteristics_max[i])

    final_predicted_numpy_actual = np.array(final_predicted_numpy_actual)
    final_predicted_numpy_actual[np.isnan(final_predicted_numpy_actual)] = 0

if z_score_normalization:
    def z_score_denormalize(arr, mean, std_dev):
        return arr * std_dev + mean
    
    final_predicted_numpy_actual = predictNextSequence.copy()
    
    # Loop over the characteristics to denormalize each one
    for i in range(number_of_characteristics):
        value = final_predicted_numpy_actual[i]
        param1, param2 = characteristics_params[i]
        final_predicted_numpy_actual[i] = z_score_denormalize(value, param1, param2)

    final_predicted_numpy_actual = np.array(final_predicted_numpy_actual)
    final_predicted_numpy_actual[np.isnan(final_predicted_numpy_actual)] = 0

# Convert predictions to appropriate format:
# Make sure we don't squeeze an axis that doesn't exist
if final_predicted_numpy_actual.ndim > 1 and final_predicted_numpy_actual.shape[1] == 1:
    array_32_80 = np.squeeze(final_predicted_numpy_actual, axis=1)
else:
    array_32_80 = final_predicted_numpy_actual  # No need to squeeze if shape is already correct

# Proceed with further processing
final_matrix = array_32_80





# Save the predicted results (e.g., for MFCC or Encodec-based predictions)
if encoding_method == "encodec":
    # PyTorch EnCodec prediction reconstruction
    #final_matrix = np.floor(final_matrix).astype(np.int64)
    #final_matrix = np.clip(final_matrix, 0,1023)  # Clamp values within valid range
    #final_matrix = 0.8*final_matrix  # Clamp values within valid range
    final_matrix = np.clip(final_matrix, 0,1023)  # Clamp values within valid range
    codes_from_numpy = torch.tensor(final_matrix, dtype=torch.long).unsqueeze(0).to('cpu')  # Add batch dimension
    with torch.no_grad():
        reconstructed_wav = model_enc.decode([(codes_from_numpy, None)])  # Decode using numpy-based codes
    # Save the reconstructed audio
    sf.write(output_file, reconstructed_wav.squeeze(0).cpu().numpy().T, model_enc.sample_rate)
    

    final_matrix_actual = codes_numpy[:,index_check+past_information_samples:index_check+past_information_samples+future_information_samples]
    codes_from_numpy = torch.tensor(final_matrix_actual, dtype=torch.long).unsqueeze(0).to('cpu')  # Add batch dimension
    with torch.no_grad():
        reconstructed_wav = model_enc.decode([(codes_from_numpy, None)])  # Decode using numpy-based codes
    # Save the reconstructed audio
    sf.write(output_file2, reconstructed_wav.squeeze(0).cpu().numpy().T, model_enc.sample_rate)
        
    
elif encoding_method == "mfcc":
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    final_matrix_filtered = final_matrix.copy()
    mel_spec = mfcc_to_mel_spectrogram(final_matrix_filtered.T)
    waveform_mfcc_based = mel_to_waveform(mel_spec, new_sampling_rate)
    waveform_mfcc_final = librosa.resample(waveform_mfcc_based, orig_sr=new_sampling_rate, target_sr=sr)    
    # Save the truncated audio
    sf.write(output_file, waveform_mfcc_final, sr)



    final_matrix_actual = mfcc[:,index_check+past_information_samples:index_check+past_information_samples+future_information_samples]
    mel_spec = mfcc_to_mel_spectrogram(final_matrix_actual.T)
    # Reconstruct waveform from Mel spectrogram
    waveform_mfcc_based = mel_to_waveform(mel_spec, new_sampling_rate)
    waveform_mfcc_final = librosa.resample(waveform_mfcc_based, orig_sr=new_sampling_rate, target_sr=sr)
    plt.figure()
    plt.plot(waveform_mfcc_final)
    plt.show()
    # Save the truncated audio
    sf.write(output_file2, waveform_mfcc_final, sr)   














predict_for_new_song = False

if predict_for_new_song:
    ############
    # predict for another song:
    # Load the new song, process it, and predict MFCC output

    audio_file_another = _path_ + r'\song2.wav'  # Specify the new song file path
    output_file_another = _path_ + r'\predicted_songx.wav'
    output_file_another_direct = _path_ + r'\direct_from_mfcc_songx.wav'
    output_file_input_samples = _path_ + r'\input_600_samples_songx.wav'  # New file to save the input 600 samples

    # Step 1: Load and preprocess the new song
    y_new, sr_new = librosa.load(audio_file_another)
    y_resampled_new = librosa.resample(y_new, orig_sr=sr_new, target_sr=new_sampling_rate)
    plt.figure()
    plt.plot(y_resampled_new)
    plt.show()

    mfcc_new = librosa.feature.mfcc(y=y_resampled_new, sr=new_sampling_rate, n_mfcc=mfcc_characteristics)
    all_notes_new = (mfcc_new.copy()).T

    # Enlarge the signal for training, if necessary
    all_notes_new = np.tile(all_notes_new, (num_duplicates, 1))

    # Convert to float
    all_notes_new = all_notes_new.astype(float)

    # Normalize using the same method as training
    if min_max_normalization:
        # Use the same min/max as original training if stored; otherwise, re-run normalization logic
        # Here we re-run a similar logic, but ideally you'd reuse characteristics_min and characteristics_max from training
        for i in range(len(all_notes_new[0])):
            all_notes_new[:, i], _, _ = normalize(all_notes_new[:, i], 0, 1)

    if z_score_normalization:
        # Use z-score normalization. We must use the same mean/std from training.
        # Here we apply a fresh normalization, but ideally we should use the stored parameters characteristics_params
        for i in range(len(all_notes_new[0])):
            mean, std = characteristics_params[i]
            all_notes_new[:, i] = (all_notes_new[:, i] - mean) / std

    # ------------------ FINE-TUNING PART ------------------
    # We'll take a small portion of the new piece for fine-tuning the model.
    # For example, we take a range of samples before our target index_check for fine-tuning.
    # Make sure index_check is large enough to allow a small training window.
    
    fine_tune_samples = 1000  # number of samples to use for fine-tuning (adjust as needed)
    if index_check < fine_tune_samples + past_information_samples + future_information_samples:
        index_check = fine_tune_samples + past_information_samples + future_information_samples

    # Prepare a fine-tuning dataset: similar logic as main training
    # We'll create input-output pairs from all_notes_new for a small training window
    start_ft = index_check - fine_tune_samples - (past_information_samples + future_information_samples)
    if start_ft < 0:
        start_ft = 0

    end_ft = start_ft + fine_tune_samples + past_information_samples + future_information_samples
    if end_ft > len(all_notes_new):
        end_ft = len(all_notes_new)

    fine_tune_segment = all_notes_new[start_ft:end_ft, :] 

    # Prepare fine-tune input/output data just like main training
    ft_num_samples = len(fine_tune_segment) - past_information_samples - future_information_samples
    if ft_num_samples < 1:
        ft_num_samples = 1  # Ensure at least one sample

    fine_tune_inputs = np.zeros((ft_num_samples, past_information_samples, number_of_characteristics))
    fine_tune_outputs = []  # list of arrays for each characteristic
    for i in range(number_of_characteristics):
        fine_tune_outputs.append(np.zeros((ft_num_samples, future_information_samples, 1)))

    for time_index in range(ft_num_samples):
        fine_tune_inputs[time_index, :, :] = fine_tune_segment[time_index:time_index+past_information_samples, :]
        for i in range(number_of_characteristics):
            out_seq = fine_tune_segment[time_index+past_information_samples:time_index+past_information_samples+future_information_samples, i]
            fine_tune_outputs[i][time_index, :, 0] = out_seq

    # Convert fine-tune data to torch tensors (assuming PyTorch model)
    if train_using_AdvancedTCN_faster:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ft_input_tensor = torch.tensor(fine_tune_inputs, dtype=torch.float32).to(device)
        # We'll fine-tune each model characteristic-by-characteristic
        # A small number of epochs and low LR
        ft_lr = 1e-2
        ft_epochs = 3  # A few epochs only

        from torch.utils.data import TensorDataset, DataLoader

        for i in range(number_of_characteristics):
            ft_output_tensor = torch.tensor(fine_tune_outputs[i], dtype=torch.float32).to(device)
            ft_dataset = TensorDataset(ft_input_tensor, ft_output_tensor)
            ft_loader = DataLoader(ft_dataset, batch_size=16, shuffle=True)

            model = loaded_models[i].to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr)
            criterion = nn.MSELoss()

            print(f"Fine-tuning model for characteristic {i} on new piece snippet...")
            for epoch in range(ft_epochs):
                train_loss = 0.0
                for inp, tgt in ft_loader:
                    optimizer.zero_grad()
                    out = model(inp)
                    # Reshape as in main training
                    out = out.unsqueeze(-1)
                    loss = criterion(out, tgt)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                print(f"Fine-tuning Epoch {epoch+1}, Loss: {train_loss/len(ft_loader)}")

            # After fine-tuning:
            model.eval()
            loaded_models[i] = model  # Update the model with the fine-tuned version

    # ---------------------------------------------------

    # Step 2 (adjusted if index_check changed): Use 600 time steps to predict the next 180
    # Now we proceed as before
    user_input_time_step = all_notes_new[index_check:index_check + past_information_samples, :]

    # Step 3: Already normalized above. user_input_time_step is normalized relative to old parameters
    normalized_user_input_time_step = np.array(user_input_time_step.copy())

    # Reshape the input to match the model’s expected input format
    normalized_user_input_time_step_reshaped = normalized_user_input_time_step.reshape((1, past_information_samples, characteristics_buff))

    # Step 4: Predict the next 180 time steps using the fine-tuned model
    predictNextSequence_new = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(0, number_of_characteristics):
        if train_using_AdvancedTCN_faster:
            test_tensor_new = torch.tensor(normalized_user_input_time_step_reshaped, dtype=torch.float32).to(device)
            model = loaded_models[i].to(device)
            model.eval()
            with torch.no_grad():
                output = model(test_tensor_new)
            output = output.cpu().numpy()
            temp_new = np.squeeze(output)[:future_information_samples]  # Predict only the next 180 time steps
            predictNextSequence_new.append(temp_new)
        else:
            # If you had a TF model, you'd just call .predict() here
            temp_new = loaded_models[i].predict(normalized_user_input_time_step_reshaped, verbose=1)
            temp_new = temp_new[:future_information_samples]
            predictNextSequence_new.append(temp_new)

    final_predicted_new_numpy = np.array(predictNextSequence_new).T

    # Step 5: Denormalize if needed
    if min_max_normalization:
        for i in range(number_of_characteristics):
            final_predicted_new_numpy[:, i] = denormalize(final_predicted_new_numpy[:, i], characteristics_min[i], characteristics_max[i])

    if z_score_normalization:
        for i in range(number_of_characteristics):
            value_new = final_predicted_new_numpy[:, i]
            param1, param2 = characteristics_params[i]
            final_predicted_new_numpy[:, i] = z_score_denormalize(value_new, param1, param2)

    # Handle invalid values
    final_predicted_new_numpy = np.nan_to_num(final_predicted_new_numpy, nan=0.0, posinf=1.0, neginf=-1.0).T

    # Step 7: Convert predicted MFCC matrix to an audio waveform
    if encoding_method == "mfcc":
        mel_spec_new = mfcc_to_mel_spectrogram(final_predicted_new_numpy.T)
        mel_spec_new = np.nan_to_num(mel_spec_new, nan=0.0, posinf=1.0, neginf=-1.0)
        waveform_predicted_new = mel_to_waveform(mel_spec_new, new_sampling_rate)
        waveform_predicted_final_new = librosa.resample(waveform_predicted_new, orig_sr=new_sampling_rate, target_sr=sr_new)
        sf.write(output_file_another, waveform_predicted_final_new, sr_new)
        plt.figure()
        plt.plot(waveform_predicted_final_new)
        plt.show()

    # Directly extract MFCC portion from the new piece (for comparison)
    final_matrix_actual_new = mfcc_new[:, index_check + past_information_samples:index_check + past_information_samples + future_information_samples]
    mel_spec_actual_new = mfcc_to_mel_spectrogram(final_matrix_actual_new.T)
    mel_spec_actual_new = np.nan_to_num(mel_spec_actual_new, nan=0.0, posinf=1.0, neginf=-1.0)
    waveform_mfcc_based_new = mel_to_waveform(mel_spec_actual_new, new_sampling_rate)
    waveform_mfcc_final_new = librosa.resample(waveform_mfcc_based_new, orig_sr=new_sampling_rate, target_sr=sr_new)
    sf.write(output_file_another_direct, waveform_mfcc_final_new, sr_new)
    plt.figure()
    plt.plot(waveform_mfcc_final_new)
    plt.show()

    # Step 8: Denormalize and save the waveform corresponding to the 600 input samples
    if min_max_normalization:
        for i in range(number_of_characteristics):
            user_input_time_step[:, i] = denormalize(user_input_time_step[:, i], characteristics_min[i], characteristics_max[i])

    if z_score_normalization:
        for i in range(number_of_characteristics):
            user_input_time_step[:, i] = z_score_denormalize(user_input_time_step[:, i], characteristics_params[i][0], characteristics_params[i][1])

    mel_spec_input = mfcc_to_mel_spectrogram(user_input_time_step)
    mel_spec_input = np.nan_to_num(mel_spec_input, nan=0.0, posinf=1.0, neginf=-1.0)
    waveform_input_samples = mel_to_waveform(mel_spec_input, new_sampling_rate)
    waveform_input_samples_resampled = librosa.resample(waveform_input_samples, orig_sr=new_sampling_rate, target_sr=sr_new)
    sf.write(output_file_input_samples, waveform_input_samples_resampled, sr_new)








predict_for_new_song2 = False
fine_tune = True
train_uppon_basic_models = True
ft_lr = 1e-4
ft_epochs = 3

if predict_for_new_song2:
    ############
    # predict for another song:
    # Load the new song, process it, and predict MFCC output

    audio_file_another = _path_ + r'\song2.wav'  # Specify the new song file path
    output_file_another = _path_ + r'\predicted_songx.wav'
    output_file_another_direct = _path_ + r'\direct_from_mfcc_songx.wav'
    output_file_input_samples = _path_ + r'\input_600_samples_songx.wav'  # New file to save the input 600 samples

    # Step 1: Load and preprocess the new song
    y_new, sr_new = librosa.load(audio_file_another)
    y_resampled_new = librosa.resample(y_new, orig_sr=sr_new, target_sr=new_sampling_rate)
    plt.figure()
    plt.plot(y_resampled_new)
    plt.show()

    mfcc_new = librosa.feature.mfcc(y=y_resampled_new, sr=new_sampling_rate, n_mfcc=mfcc_characteristics)
    all_notes_new = (mfcc_new.copy()).T

    # Enlarge the signal for training, if necessary
    all_notes_new = np.tile(all_notes_new, (num_duplicates, 1))

    # Convert to float
    all_notes_new = all_notes_new.astype(float)

    # Normalize using the same method as training
    if min_max_normalization:
        for i in range(len(all_notes_new[0])):
            all_notes_new[:, i], _, _ = normalize(all_notes_new[:, i], 0, 1)

    if z_score_normalization:
        for i in range(len(all_notes_new[0])):
            mean, std = characteristics_params[i]
            all_notes_new[:, i] = (all_notes_new[:, i] - mean) / std

    # ------------------ FINE-TUNING PART ------------------
    fine_tune_samples = 1000  # number of samples to use for fine-tuning
    if index_check < fine_tune_samples + past_information_samples + future_information_samples:
        index_check = fine_tune_samples + past_information_samples + future_information_samples

    start_ft = index_check - fine_tune_samples - (past_information_samples + future_information_samples)
    if start_ft < 0:
        start_ft = 0

    end_ft = start_ft + fine_tune_samples + past_information_samples + future_information_samples
    if end_ft > len(all_notes_new):
        end_ft = len(all_notes_new)

    fine_tune_segment = all_notes_new[start_ft:end_ft, :] 

    ft_num_samples = len(fine_tune_segment) - past_information_samples - future_information_samples
    if ft_num_samples < 1:
        ft_num_samples = 1

    fine_tune_inputs = np.zeros((ft_num_samples, past_information_samples, number_of_characteristics))
    fine_tune_outputs = []
    for i in range(number_of_characteristics):
        fine_tune_outputs.append(np.zeros((ft_num_samples, future_information_samples, 1)))

    for time_index in range(ft_num_samples):
        fine_tune_inputs[time_index, :, :] = fine_tune_segment[time_index:time_index+past_information_samples, :]
        for i in range(number_of_characteristics):
            out_seq = fine_tune_segment[time_index+past_information_samples:time_index+past_information_samples+future_information_samples, i]
            fine_tune_outputs[i][time_index, :, 0] = out_seq

    # Convert fine-tune data to torch tensors (assuming PyTorch model)
    
    if train_using_AdvancedTCN_faster and fine_tune == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ft_input_tensor = torch.tensor(fine_tune_inputs, dtype=torch.float32).to(device)


        from torch.utils.data import TensorDataset, DataLoader


    # If we start from scratch (train_uppon_basic_models=False), create fresh models
        if not train_uppon_basic_models:
            loaded_models = []
            for i in range(number_of_characteristics):
                model = AdvancedTCN_faster(num_channels=[128, 512, 1024], kernel_size=5, output_size=future_information_samples, dilations=[1, 4, 8],input_channels = len(all_notes[0])).to(device)
                loaded_models.append(model)

        for i in range(number_of_characteristics):
            ft_output_tensor = torch.tensor(fine_tune_outputs[i], dtype=torch.float32).to(device)
            ft_dataset = TensorDataset(ft_input_tensor, ft_output_tensor)
            ft_loader = DataLoader(ft_dataset, batch_size=16, shuffle=True)

            model = loaded_models[i].to(device)

            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr)
            criterion = nn.MSELoss()

            print(f"Fine-tuning model for characteristic {i} on new piece snippet...")
            for epoch in range(ft_epochs):
                train_loss = 0.0
                for inp, tgt in ft_loader:
                    optimizer.zero_grad()
                    out = model(inp)
                    out = out.unsqueeze(-1)
                    loss = criterion(out, tgt)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                print(f"Fine-tuning Epoch {epoch+1}, Loss: {train_loss/len(ft_loader)}")

            model.eval()
            loaded_models[i] = model

    # ---------------------------------------------------

    # Step 2: Use 600 time steps to predict the next 180
    user_input_time_step = all_notes_new[index_check:index_check + past_information_samples, :]

    # Already normalized above
    normalized_user_input_time_step = np.array(user_input_time_step.copy())
    normalized_user_input_time_step_reshaped = normalized_user_input_time_step.reshape((1, past_information_samples, characteristics_buff))

    # Step 4: Predict the next 180 time steps using the fine-tuned model
    predictNextSequence_new = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(0, number_of_characteristics):
        if train_using_AdvancedTCN_faster:
            test_tensor_new = torch.tensor(normalized_user_input_time_step_reshaped, dtype=torch.float32).to(device)
            model = loaded_models[i].to(device)
            model.eval()
            with torch.no_grad():
                output = model(test_tensor_new)
            output = output.cpu().numpy()
            temp_new = np.squeeze(output)[:future_information_samples]
            predictNextSequence_new.append(temp_new)
        else:
            temp_new = loaded_models[i].predict(normalized_user_input_time_step_reshaped, verbose=1)
            temp_new = temp_new[:future_information_samples]
            predictNextSequence_new.append(temp_new)

    final_predicted_new_numpy = np.array(predictNextSequence_new).T

    # Step 5: Denormalize if needed
    if min_max_normalization:
        for i in range(number_of_characteristics):
            final_predicted_new_numpy[:, i] = denormalize(final_predicted_new_numpy[:, i], characteristics_min[i], characteristics_max[i])

    if z_score_normalization:
        for i in range(number_of_characteristics):
            value_new = final_predicted_new_numpy[:, i]
            param1, param2 = characteristics_params[i]
            final_predicted_new_numpy[:, i] = z_score_denormalize(value_new, param1, param2)

    final_predicted_new_numpy = np.nan_to_num(final_predicted_new_numpy, nan=0.0, posinf=1.0, neginf=-1.0).T

    # Step 6: Convert predicted MFCC to Mel spectrogram
    mel_spec_new = mfcc_to_mel_spectrogram(final_predicted_new_numpy.T)
    mel_spec_new = np.nan_to_num(mel_spec_new, nan=0.0, posinf=1.0, neginf=-1.0)

    # ----------- SMOOTHING THE MEL SPECTROGRAM -----------
    from scipy.signal import savgol_filter

    def smooth_spectrogram(mel_spec, window_length=7, polyorder=2):
        mel_spec_smooth = np.copy(mel_spec)
        for i in range(mel_spec_smooth.shape[0]):
            mel_spec_smooth[i, :] = savgol_filter(mel_spec_smooth[i, :], window_length=window_length, polyorder=polyorder)
        return mel_spec_smooth



    mel_spec_new_smoothed = smooth_spectrogram(mel_spec_new, window_length=7, polyorder=2)
    #mel_spec_new_smoothed = mel_spec_new
    
    
    # ----------- USE GRIFFIN-LIM OR A VOCODER -----------
    use_vocoder = False  # Set to True if you have a pretrained vocoder
    if use_vocoder:
        # Example: Using a pretrained HiFi-GAN vocoder
        # You must have the model and configs available.
        # Download a pretrained vocoder from HiFi-GAN GitHub or other sources.
        # Pseudo-code (adapt this to your environment):
        from models import Generator  # Adjust based on your HiFi-GAN code
        import torch

        class AttrDict(dict):
            __getattr__ = dict.__getitem__

        # Example HiFi-GAN config (adjust to match your model)
        hifigan_config = {
            "resblock_kernel_sizes": [3,7,11],
            "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
            "upsample_rates": [8,8,2,2],
            "upsample_kernel_sizes": [16,16,4,4],
            "upsample_initial_channel": 512,
            "gin_channels": 0,
            "segment_size": 8192,
            "num_mels": mel_spec_new_smoothed.shape[0]
        }

        h = AttrDict(hifigan_config)
        generator = Generator(h).to('cpu')
        generator.load_state_dict(torch.load("hifigan_generator.pth", map_location='cpu')['generator'])
        generator.eval()
        generator.remove_weight_norm()

        def hifigan_infer(mel):
            # Adjust normalization to match what HiFi-GAN expects
            # For example, if HiFi-GAN trained on log-mels scaled between -1 and 1:
            # mel = (mel - mean) / std  (depends on your training)
            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0) # (1, n_mels, n_frames)
            with torch.no_grad():
                y_g_hat = generator(x)
            audio = y_g_hat.squeeze().cpu().numpy()
            return audio

        waveform_predicted_new_vocoded = hifigan_infer(mel_spec_new_smoothed)
        waveform_predicted_final_new = librosa.resample(waveform_predicted_new_vocoded, orig_sr=new_sampling_rate, target_sr=sr_new)

        sf.write(output_file_another, waveform_predicted_final_new, sr_new)
        plt.figure()
        plt.plot(waveform_predicted_final_new)
        plt.show()
    else:
        # If not using a vocoder, use Griffin-Lim with more iterations for smoother result
        waveform_predicted_new = librosa.feature.inverse.mel_to_audio(mel_spec_new_smoothed,
                                                                      sr=new_sampling_rate,
                                                                      n_iter=200)  # More iterations
        waveform_predicted_final_new = librosa.resample(waveform_predicted_new, orig_sr=new_sampling_rate, target_sr=sr_new)
        sf.write(output_file_another, waveform_predicted_final_new, sr_new)
        plt.figure()
        plt.plot(waveform_predicted_final_new)
        plt.show()

    # Directly extract MFCC portion from the new piece (for comparison)
    final_matrix_actual_new = mfcc_new[:, index_check + past_information_samples:index_check + past_information_samples + future_information_samples]
    mel_spec_actual_new = mfcc_to_mel_spectrogram(final_matrix_actual_new.T)
    mel_spec_actual_new = np.nan_to_num(mel_spec_actual_new, nan=0.0, posinf=1.0, neginf=-1.0)

    waveform_mfcc_based_new = mel_to_waveform(mel_spec_actual_new, new_sampling_rate)
    waveform_mfcc_final_new = librosa.resample(waveform_mfcc_based_new, orig_sr=new_sampling_rate, target_sr=sr_new)
    sf.write(output_file_another_direct, waveform_mfcc_final_new, sr_new)
    plt.figure()
    plt.plot(waveform_mfcc_final_new)
    plt.show()

    # Step 8: Denormalize and save the waveform corresponding to the 600 input samples
    if min_max_normalization:
        for i in range(number_of_characteristics):
            user_input_time_step[:, i] = denormalize(user_input_time_step[:, i], characteristics_min[i], characteristics_max[i])

    if z_score_normalization:
        for i in range(number_of_characteristics):
            user_input_time_step[:, i] = z_score_denormalize(user_input_time_step[:, i], characteristics_params[i][0], characteristics_params[i][1])

    mel_spec_input = mfcc_to_mel_spectrogram(user_input_time_step)
    mel_spec_input = np.nan_to_num(mel_spec_input, nan=0.0, posinf=1.0, neginf=-1.0)
    waveform_input_samples = mel_to_waveform(mel_spec_input, new_sampling_rate)
    waveform_input_samples_resampled = librosa.resample(waveform_input_samples, orig_sr=new_sampling_rate, target_sr=sr_new)
    sf.write(output_file_input_samples, waveform_input_samples_resampled, sr_new)







crosssynthesis = False
if crosssynthesis == True:
    # Also define a stable source audio for cross-synthesis:
    high_quality_source = _path_ + r'\song2.wav'  # A stable, rich sounding music file


    # ---------------- CROSS-SYNTHESIS EXAMPLE ----------------
    # Now we combine waveform_predicted_final_new with a stable source
    # We'll take STFT of both and use predicted magnitude with source phase

    # Load the stable source
    stable_source_audio, sr_source = librosa.load(high_quality_source, sr=sr_new)  # resample to sr_new for consistency

    # Ensure stable source is long enough or trim/pad
    len_pred = len(waveform_predicted_final_new)
    if len(stable_source_audio) < len_pred:
        # Pad stable source
        stable_source_audio = np.pad(stable_source_audio, (0, len_pred - len(stable_source_audio)), mode='reflect')
    else:
        stable_source_audio = stable_source_audio[400000:400000+len_pred]

    # STFT parameters
    n_fft = 2048
    hop_length = 512
    win_length = 2048

    # STFT of predicted
    D_pred = librosa.stft(waveform_predicted_final_new, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitude_pred, phase_pred = np.abs(D_pred), np.angle(D_pred)

    # STFT of stable source
    D_source = librosa.stft(stable_source_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitude_source, phase_source = np.abs(D_source), np.angle(D_source)

    # Cross-synthesis: Use predicted magnitude and stable source phase
    D_hybrid = 2*magnitude_pred*magnitude_source * np.exp(1j * phase_source *phase_pred)

    # Inverse STFT
    waveform_hybrid = librosa.istft(D_hybrid, hop_length=hop_length, win_length=win_length)

    # Save and plot the hybrid result
    output_file_hybrid = _path_ + r'\predicted_songx_hybrid.wav'
    sf.write(output_file_hybrid, waveform_hybrid, sr_new)
    plt.figure()
    plt.plot(waveform_hybrid)
    plt.title("Hybrid Waveform (Cross-Synthesis)")
    plt.show()

    # This hybrid audio should have the timing and rough spectral shape from predicted,
    # but phase and hopefully some timbral quality from the stable source.






