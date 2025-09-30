

_path_ = r"/home/alels_star/Desktop/ekpa_diplom/new3"



import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.stats import skew, kurtosis
#from encodec import EncodecModel
#from encodec.utils import convert_audio
import torchaudio
import torch
import soundfile as sf
import time
from playsound import playsound
from scipy.signal import savgol_filter
import scipy.signal as sig
import glob
import gc
import tensorflow as tf
print(tf. __version__)
import keras
#tf.compat.v1.disable_eager_execution()
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.layers import BatchNormalization    
import tensorflow as tf
from timeit import default_timer as timer




#########################################   If using mfcc:
encoding_method="cqt"
mfcc_characteristics = 110
new_sampling_rate = 41000

#########################################   If using encodec:
#encoding_method="encodec"
encodec_bandwidth = 6.0 # For example, a bandwidth of 24 corresponds to 32 characteristics
#############################################################


model_num = 9
model_name = "AImodel"+str(model_num)

if encoding_method=="mfcc":
    past_information_samples = 900   #No of samples used as knowledge
    future_information_samples = 260   #No of samples to predict in the future
    epochs_number = 1
elif encoding_method == "cqt":
    past_information_samples = 1230   #No of samples used as knowledge
    future_information_samples = 500   #No of samples to predict in the future
    epochs_number = 8
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

retrain=True
#############################################################

'''
END USER PARAMETERS
'''


# uder defined:
if True:
    print("matching model")
    ## Create models subfolder  1st AI
    # checking if the directory exist or not. 
    if not os.path.exists(_path_ + r"/" + model_name):      
        # if the directory is not present  then create it. 
        os.makedirs(_path_ + r"/" + model_name) 
        
    
    # Directory containing all songs
    songs_dir = os.path.join(_path_, 'training_songs_model_'+str(model_num))
    print("training songs dir:", songs_dir)
    
    
    

# Get a list of all .wav files in the songs directory
audio_files = glob.glob(os.path.join(songs_dir, '*.wav'))

# Initialize a list to hold features from all songs
all_notes_list = []

for audio_file in audio_files:
    print(f"Processing {audio_file}...")
    if encoding_method == "mfcc":
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


    elif encoding_method == "cqt":
        # Load and preprocess audio
        y, sr = librosa.load(audio_file)
        y2 = librosa.resample(y, orig_sr=sr, target_sr=new_sampling_rate)


        train_with_GPU_and_CPU_RAM = False
        train_with_only_GPU_RAM = True
        # Compute the Constant-Q Transform
        if model_num == 7:
            cqt = librosa.cqt(y2, sr=new_sampling_rate, hop_length=200, n_bins=50)# default n_bins=84, 56 is good enough
        if model_num == 8:
            cqt = librosa.cqt(y2, sr=new_sampling_rate, hop_length=200, n_bins=65)# default n_bins=84, 56 is good enough
        if model_num == 9:
            cqt = librosa.cqt(y2, sr=new_sampling_rate, hop_length=190, n_bins=50)# default n_bins=84, 56 is good enough

        # Prepare "all_notes" like MFCC version: (Time Steps, Features)
        #all_notes = cqt.T.copy()

        # Split into real and imaginary parts
        cqt_real = cqt.real.T  # Shape: (Time Steps, Freq Bins)
        cqt_imag = cqt.imag.T  # Shape: (Time Steps, Freq Bins)
        
        # Concatenate along the feature dimension
        all_notes = np.concatenate([cqt_real, cqt_imag], axis=1)  # Shape: (Time Steps, 2 * Freq Bins)

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
np.savetxt(_path_+r'/summary_notes_audio.csv', all_notes, delimiter=',')

# Load and print the normalized data to verify
datax = np.loadtxt(_path_+r'/summary_notes_audio.csv', delimiter=',')
#datax = datax.astype(int)

print_characteristics = False
if print_characteristics == True:
    if min_max_normalization == True:
        print("characteristics_max",characteristics_max)
        print("characteristics_min",characteristics_min)
    if z_score_normalization == True:
        print("characteristics_params",characteristics_params)    


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


    starting_characteristic = 0
    #starting_characteristic = 112
    
    # Loop through characteristics to train the models
    for i in range(starting_characteristic, number_of_characteristics):
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
            path = _path_ + r"/"+model_name+"//"
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



            if train_with_only_GPU_RAM ==True:
                    if i == starting_characteristic:
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

            if train_with_GPU_and_CPU_RAM == True:
                    if i == starting_characteristic:
                        # Convert input data and output data to PyTorch tensors
                        #input_tensor = torch.tensor(input_data1, dtype=torch.float32).to(device)
                        input_tensor = torch.tensor(input_data1, dtype=torch.float32)#.to(device)
                    #output_tensor = torch.tensor(output_data1[i], dtype=torch.float32).to(device)
                    output_tensor = torch.tensor(output_data1[i], dtype=torch.float32)#.to(device)
                    
                    
                    
                    # Create a TensorDataset and DataLoader
                    dataset = TensorDataset(input_tensor, output_tensor)
                    #train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
                    train_loader = DataLoader(
                        dataset,
                        batch_size=256,
                        shuffle=True,
                        pin_memory=True  # speeds up CPU->GPU transfers
                    )
                    print("No of batches: ")
                    print(len(train_loader))
        
        
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
                            #inputs, targets = inputs.to(device), targets.to(device)
                            inputs, targets = inputs.pin_memory(), targets.pin_memory()
                            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
                    
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
            path = _path_ + r"/"+model_name+"//"
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
    for i in range(starting_characteristic, number_of_characteristics):
        if any([train_using_LSTM, train_using_TCN, train_using_transformer, train_using_transformer2, train_using_advanced_TCN]):
            # Load TensorFlow model
            path = _path_ + r"/"+model_name+"//"
            loaded_models.append(tf.keras.models.load_model(path + f"custom_model_{i}.keras"))
        elif train_using_AdvancedTCN_faster == True:
            # Load PyTorch model
            # model = AdvancedTCN_faster(num_channels=[64, 128, 256], kernel_size=3, output_size=future_information_samples, dilations=[1, 2, 4],input_channels = len(all_notes[0]))
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AdvancedTCN_faster(num_channels=[128, 512, 1024], kernel_size=5, output_size=future_information_samples, dilations=[1, 4, 8],input_channels = len(all_notes[0]))
            path = _path_ + r"/"+model_name+"//"
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
            path = _path_ + r"/"+model_name+"//"
            loaded_models.append(tf.keras.models.load_model(path + f"custom_model_{i}.keras"))
        elif train_using_AdvancedTCN_faster == True:
            # Load PyTorch model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # model = AdvancedTCN_faster(num_channels=[64, 128, 256], kernel_size=3, output_size=future_information_samples, dilations=[1, 2, 4],input_channels = len(all_notes[0]))
            model = AdvancedTCN_faster(num_channels=[128, 512, 1024], kernel_size=5, output_size=future_information_samples, dilations=[1, 4, 8],input_channels = len(all_notes[0]))
            path = _path_ + r"/"+model_name+"//"
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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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




#%%

# Everything here remains as you pasted it, except we'll add a new button and function
# for RAVE. The rest of the snippet (model inference, filter code, etc.) is untouched.



###############################################################################
# Here are your placeholders / global definitions
###############################################################################
filter_mfcc_approach = True
filter_mel_approach1 = True

def normalize(x, min_val, max_val):
    x_min, x_max = np.min(x), np.max(x)
    denom = (x_max - x_min) if (x_max - x_min)!=0 else 1e-8
    x_norm = (x - x_min) * (max_val - min_val) / denom
    return x_norm, x_min, x_max

def denormalize(x_norm, old_min, old_max):
    return x_norm * (old_max - old_min) + old_min

def z_score_normalize(x):
    mean_val, std_val = np.mean(x), np.std(x)
    if std_val == 0:
        std_val = 1e-8
    x_norm = (x - mean_val) / std_val
    return x_norm, mean_val, std_val

def z_score_denormalize(x_norm, mean_val, std_val):
    return x_norm * std_val + mean_val

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

