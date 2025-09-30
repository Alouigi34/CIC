
import os

# ─── Read & parse the env vars ────────────────────────────────────
chosen_model = os.environ["CHOSEN_MODEL"]

# these were accidentally tuples before; now they're plain strings
_path_                    = os.environ["_path_"]
audio_file_another        = os.environ["audio_file_another"]
models_path               = os.environ["models_path"]
songs_path_                = os.environ.get("songs_path", "")
user_defined_time_prediction_index=os.environ["user_defined_time_prediction_index"],


index_check      = int(os.environ.get("index_check", "0"))
length_percentage= float(os.environ.get("length_percentage", "1.0"))

# ─── Fix Matplotlib for headless / batch mode ────────────────────
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("Agg")

# ─── Debug / sanity printout ──────────────────────────────────────
print("=== ENVIRONMENT SETTINGS ===")
print(f" Model:   {chosen_model}")
print(f" _path_:  { _path_ }")
print(f" Audio:   { audio_file_another }")
print(f" Models:  { models_path }")
print(f" Songs:   { songs_path_ }")
print(f" UseIdx:  { user_defined_time_prediction_index }")
print(f" IdxChk:  { index_check }")
print(f" Length%: { length_percentage }")
print("=============================")




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
#from playsound import playsound
from scipy.signal import savgol_filter
import scipy.signal as sig
import glob
import gc
import tensorflow as tf
print(tf. __version__)
import keras

# Patch get_custom_objects (if not already patched)
keras.saving.get_custom_objects = tf.keras.utils.get_custom_objects

# Patch register_keras_serializable to point to the correct function in tf.keras.utils
keras.saving.register_keras_serializable = tf.keras.utils.register_keras_serializable

#tf.compat.v1.disable_eager_execution()
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.layers import BatchNormalization    
import tensorflow as tf
from timeit import default_timer as timer
output_file_input_initial = os.path.join(_path_, f"input_song.wav")





retrain=False
#############################################################

'''
END USER PARAMETERS
'''


if chosen_model == "model1":
        model_num = 1
        model_name = "AImodel"+str(model_num)
        print("matching model")
        ## Create models subfolder  1st AI
        # checking if the directory exist or not. 
        if not os.path.exists(_path_ +r"/AImodel1"):      
            # if the directory is not present  then create it. 
            os.makedirs(_path_ +r"/AImodel1") 
            
        
        # Directory containing all songs
        songs_dir = os.path.join(songs_path_, 'training_songs_model_1')
        print("training songs dir:", songs_dir)
            
        #########################################   If using mfcc:
        encoding_method="mfcc"
        mfcc_characteristics = 110
        new_sampling_rate = 41000
        
        #########################################   If using encodec:
        #encoding_method="encodec"
        encodec_bandwidth = 6.0 # For example, a bandwidth of 24 corresponds to 32 characteristics
        #############################################################
        
        if encoding_method=="mfcc":
            past_information_samples = 900   #No of samples used as knowledge
            future_information_samples = 260   #No of samples to predict in the future
            epochs_number = 10
            number_of_characteristics = mfcc_characteristics
            num_duplicates = 1
            characteristics_buff=number_of_characteristics
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
    
if chosen_model == "model2":
        model_num = 2
        model_name = "AImodel"+str(model_num)
        print("matching model")
        ## Create models subfolder  1st AI
        # checking if the directory exist or not. 
        if not os.path.exists(_path_ +r"/AImodel2"):      
            # if the directory is not present  then create it. 
            os.makedirs(_path_ +r"/AImodel2") 
            
        
        # Directory containing all songs
        songs_dir = os.path.join(songs_path_, 'training_songs_model_2')
        print("training songs dir:", songs_dir)


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
            number_of_characteristics = mfcc_characteristics
            num_duplicates = 1
            characteristics_buff=number_of_characteristics
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
    
    

if chosen_model == "model3":
        model_num = 3
        model_name = "AImodel"+str(model_num)
        print("matching model")
        ## Create models subfolder  1st AI
        # checking if the directory exist or not. 
        if not os.path.exists(_path_ +r"/AImodel3"):      
            # if the directory is not present  then create it. 
            os.makedirs(_path_ +r"/AImodel3") 
            
        
        # Directory containing all songs
        songs_dir = os.path.join(songs_path_, 'training_songs_model_3')
        print("training songs dir:", songs_dir)
        
        #########################################   If using mfcc:
        encoding_method="mfcc"
        mfcc_characteristics = 110
        new_sampling_rate = 41000
        
        #########################################   If using encodec:
        #encoding_method="encodec"
        encodec_bandwidth = 6.0 # For example, a bandwidth of 24 corresponds to 32 characteristics
        #############################################################
        
        if encoding_method=="mfcc":
            past_information_samples = 900   #No of samples used as knowledge
            future_information_samples = 260   #No of samples to predict in the future
            epochs_number = 10
            number_of_characteristics = mfcc_characteristics
            num_duplicates = 1
            characteristics_buff=number_of_characteristics
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
    
  
if chosen_model == "model4":
        model_num = 4
        model_name = "AImodel"+str(model_num)
        print("matching model")
        ## Create models subfolder  1st AI
        # checking if the directory exist or not. 
        if not os.path.exists(_path_ +r"/AImodel4"):      
            # if the directory is not present  then create it. 
            os.makedirs(_path_ +r"/AImodel4") 
            
        
        # Directory containing all songs
        songs_dir = os.path.join(songs_path_, 'training_songs_model_4')
        print("training songs dir:", songs_dir)
        
        #########################################   If using mfcc:
        encoding_method="mfcc"
        mfcc_characteristics = 110
        new_sampling_rate = 41000
        
        #########################################   If using encodec:
        #encoding_method="encodec"
        encodec_bandwidth = 6.0 # For example, a bandwidth of 24 corresponds to 32 characteristics
        #############################################################
        
        if encoding_method=="mfcc":
            past_information_samples = 900   #No of samples used as knowledge
            future_information_samples = 260   #No of samples to predict in the future
            epochs_number = 3
            number_of_characteristics = mfcc_characteristics
            num_duplicates = 1
            characteristics_buff=number_of_characteristics
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
    

  
if chosen_model == "model5":
        model_num = 5
        model_name = "AImodel"+str(model_num)
        print("matching model")
        ## Create models subfolder  1st AI
        # checking if the directory exist or not. 
        if not os.path.exists(_path_ +r"/AImodel5"):      
            # if the directory is not present  then create it. 
            os.makedirs(_path_ +r"/AImodel5") 
            
        
        # Directory containing all songs
        songs_dir = os.path.join(songs_path_, 'training_songs_model_5')
        print("training songs dir:", songs_dir)
        
        #########################################   If using mfcc:
        encoding_method="mfcc"
        mfcc_characteristics = 110
        new_sampling_rate = 41000
        
        #########################################   If using encodec:
        #encoding_method="encodec"
        encodec_bandwidth = 6.0 # For example, a bandwidth of 24 corresponds to 32 characteristics
        #############################################################
        
        if encoding_method=="mfcc":
            past_information_samples = 900   #No of samples used as knowledge
            future_information_samples = 260   #No of samples to predict in the future
            epochs_number = 1
            number_of_characteristics = mfcc_characteristics
            num_duplicates = 1
            characteristics_buff=number_of_characteristics
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
    
      

  
if chosen_model == "model6":
        model_num = 6
        model_name = "AImodel"+str(model_num)
        print("matching model")
        ## Create models subfolder  1st AI
        # checking if the directory exist or not. 
        if not os.path.exists(_path_ +r"/AImodel6"):      
            # if the directory is not present  then create it. 
            os.makedirs(_path_ +r"/AImodel6") 
            
        
        # Directory containing all songs
        songs_dir = os.path.join(songs_path_, 'training_songs_model_6')
        print("training songs dir:", songs_dir)
        
        #########################################   If using mfcc:
        encoding_method="cqt"
        new_sampling_rate = 41000
        
        #########################################   If using encodec:
        #encoding_method="encodec"
        encodec_bandwidth = 6.0 # For example, a bandwidth of 24 corresponds to 32 characteristics
        #############################################################
        
        if True:
            past_information_samples = 1260   #No of samples used as knowledge
            future_information_samples = 460   #No of samples to predict in the future
            epochs_number = 11
            number_of_characteristics = 100
            num_duplicates = 1
            characteristics_buff=number_of_characteristics
            hop_length = 200
            n_freq_bins = 50  # Original number of frequency bins in CQT
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
    
      

if chosen_model == "model7":
        model_num = 7
        model_name = "AImodel"+str(model_num)
        print("matching model")
        ## Create models subfolder  1st AI
        # checking if the directory exist or not. 
        if not os.path.exists(_path_ +r"/AImodel7"):      
            # if the directory is not present  then create it. 
            os.makedirs(_path_ +r"/AImodel7") 
            
        
        # Directory containing all songs
        songs_dir = os.path.join(songs_path_, 'training_songs_model_7')
        print("training songs dir:", songs_dir)
        
        #########################################   If using mfcc:
        encoding_method="cqt"
        new_sampling_rate = 41000
        
        #########################################   If using encodec:
        #encoding_method="encodec"
        encodec_bandwidth = 6.0 # For example, a bandwidth of 24 corresponds to 32 characteristics
        #############################################################
        
        if True:
            past_information_samples = 1260   #No of samples used as knowledge
            future_information_samples = 460   #No of samples to predict in the future
            epochs_number = 16
            number_of_characteristics = 100
            num_duplicates = 1
            characteristics_buff=number_of_characteristics
            hop_length = 200
            n_freq_bins = 50  # Original number of frequency bins in CQT
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
    
      

if chosen_model == "model8":
        model_num = 8
        model_name = "AImodel"+str(model_num)
        print("matching model")
        ## Create models subfolder  1st AI
        # checking if the directory exist or not. 
        if not os.path.exists(_path_ +r"/AImodel8"):      
            # if the directory is not present  then create it. 
            os.makedirs(_path_ +r"/AImodel8") 
            
        
        # Directory containing all songs
        songs_dir = os.path.join(songs_path_, 'training_songs_model_8')
        print("training songs dir:", songs_dir)
        
        #########################################   If using mfcc:
        encoding_method="cqt"
        new_sampling_rate = 41000
        
        #########################################   If using encodec:
        #encoding_method="encodec"
        encodec_bandwidth = 6.0 # For example, a bandwidth of 24 corresponds to 32 characteristics
        #############################################################
        
        if True:
            past_information_samples = 1260   #No of samples used as knowledge
            future_information_samples = 460   #No of samples to predict in the future
            epochs_number = 16
            number_of_characteristics = 130
            num_duplicates = 1
            characteristics_buff=number_of_characteristics
            hop_length = 200
            n_freq_bins = 65  # Original number of frequency bins in CQT

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
    

   

if chosen_model == "model9":
        model_num = 9
        model_name = "AImodel"+str(model_num)
        print("matching model")
        ## Create models subfolder  1st AI
        # checking if the directory exist or not. 
        if not os.path.exists(_path_ +r"/AImodel9"):      
            # if the directory is not present  then create it. 
            os.makedirs(_path_ +r"/AImodel9") 
            
        
        # Directory containing all songs
        songs_dir = os.path.join(songs_path_, 'training_songs_model_9')
        print("training songs dir:", songs_dir)
        
        #########################################   If using mfcc:
        encoding_method="cqt"
        new_sampling_rate = 41000
        
        #########################################   If using encodec:
        #encoding_method="encodec"
        encodec_bandwidth = 6.0 # For example, a bandwidth of 24 corresponds to 32 characteristics
        #############################################################
        
        if True:
            past_information_samples = 1230   #No of samples used as knowledge
            future_information_samples = 500   #No of samples to predict in the future
            epochs_number = 8
            number_of_characteristics = 100
            num_duplicates = 1
            characteristics_buff=number_of_characteristics
            hop_length = 190
            n_freq_bins = 50  # Original number of frequency bins in CQT

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
    
      

      
if encoding_method == "mfcc":            
        def mfcc_to_mel_spectrogram(mfcc):
            mel_spec = librosa.feature.inverse.mfcc_to_mel(mfcc.T)
            return mel_spec

        def mel_to_waveform(mel_spec, sr):
            return librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr, n_iter=280)
    




normalization_based_on_training_songs = True


if normalization_based_on_training_songs == True:
    def run_normalization():    
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
            
                    # Compute the Constant-Q Transform
                    cqt = librosa.cqt(y2, sr=new_sampling_rate, hop_length=hop_length, n_bins=n_freq_bins)# default n_bins=84, 56 is good enough
            
            
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
                
                return characteristics_max, characteristics_min
            
            
            
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
                
                return characteristics_params







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







################################################################################   
################################################################################
################################################################################   


if retrain == False:
    loaded_models = []
    for i in range(0, number_of_characteristics):
        if any([train_using_LSTM, train_using_TCN, train_using_transformer, train_using_transformer2, train_using_advanced_TCN]):
            # Load TensorFlow model
            path = models_path + r"/"+model_name+"//"
            loaded_models.append(tf.keras.models.load_model(path + f"custom_model_{i}.keras"))
        elif train_using_AdvancedTCN_faster == True:
            # Load PyTorch model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # model = AdvancedTCN_faster(num_channels=[64, 128, 256], kernel_size=3, output_size=future_information_samples, dilations=[1, 2, 4],input_channels = len(all_notes[0]))
            model = AdvancedTCN_faster(num_channels=[128, 512, 1024], kernel_size=5, output_size=future_information_samples, dilations=[1, 4, 8],input_channels = number_of_characteristics)
            path = models_path  + r"/"+model_name+"//"
            model.load_state_dict(torch.load(path + f"custom_model_{i}.pth", map_location=device))
            model = model.to(device)
            loaded_models.append(model)

   

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

























###############################################################################
# The EXACT snippet you said is working
###############################################################################
def run_inference(_path_, audio_file_another, user_defined_time_prediction_index, index_check, length_percentage):
    """
    Reproduces your exact snippet line-by-line, except:
      - _path_ is passed as an argument
      - audio_file_another is a full path to the .wav
      - user_defined_time_prediction_index is a boolean
      - index_check is an integer
    Returns: A list of .wav files created
    """

    # We'll keep track of the generated output files here:
    generated_files = []

    # EXACT SNIPPET START (minus the top-level "predict_for_new_song" check)
    # ----------------------------------------------------------------------
    y_new, sr_new = librosa.load(audio_file_another)
    y_resampled_new = librosa.resample(y_new, orig_sr=sr_new, target_sr=new_sampling_rate)
    plt.figure()
    plt.plot(y_resampled_new)
    plt.show()



    if encoding_method == "mfcc":    
        
        
            mfcc_new = librosa.feature.mfcc(y=y_resampled_new, sr=new_sampling_rate, n_mfcc=mfcc_characteristics)
            
            # Prepare the input matrix for the new song
            all_notes_new = (mfcc_new.copy()).T
            
            # Enlarge the signal for training, if necessary
            all_notes_new = np.tile(all_notes_new, (num_duplicates, 1))
            
            # Convert to float
            all_notes_new = all_notes_new.astype(float)
            
            
            
            if normalization_based_on_training_songs == True:
                if min_max_normalization:
                    characteristics_max,characteristics_min = run_normalization()
                    for i in range(len(all_notes_new[0])):
                        all_notes_new[:, i], _, _ = normalize(all_notes_new[:, i], 0, 1)
                
                if z_score_normalization:
                    characteristics_params = run_normalization()
                    for i in range(len(all_notes_new[0])):
                        all_notes_new[:, i], _, _ = z_score_normalize(all_notes_new[:, i])
     
     
                    
            if normalization_based_on_training_songs == False:        
                if min_max_normalization:                   
                    characteristics_max = []
                    characteristics_min = []
                    
                    for i in range(len(all_notes_new[0])):
                        range_to_normalize = (0, 1)
                        all_notes_new[:, i], characteristic_max, characteristic_min = normalize(all_notes_new[:, i],
                                                                                            range_to_normalize[0],
                                                                                            range_to_normalize[1])
                        characteristics_max.append(characteristic_max)
                        characteristics_min.append(characteristic_min)                   
                                    
                        
    
                if z_score_normalization:
                    characteristics_params = []
                    for i in range(len(all_notes_new[0])):
                        #all_notes_new[:, i], _, _ = z_score_normalize(all_notes_new[:, i])
                        all_notes_new[:, i], param1, param2  = z_score_normalize(all_notes_new[:, i])
                        characteristics_params.append((param1, param2))
                
                           
            
    
 
            # user_defined_time_prediction_index logic
            if user_defined_time_prediction_index and len(all_notes_new) > index_check + past_information_samples:
                # Take 600 time steps from index_check
                user_input_time_step = all_notes_new[index_check:index_check + past_information_samples, :]
                
                # We replicate exactly how the snippet constructs these paths
                # output_file_another, output_file_another_direct, output_file_input_samples
                output_file_another = os.path.join(_path_, f"predicted_song_{index_check}.wav")
                output_file_another_direct = os.path.join(_path_, f"direct_from_mfcc_song_{index_check}.wav")
                output_file_input_samples = os.path.join(_path_, f"input_buffer_mfcc_samples_song_{index_check}.wav")
                
            else:
                # user_defined_time_prediction_index = False
                # or data is not large enough
                index_check = len(all_notes_new)
                user_input_time_step = all_notes_new[-past_information_samples:, :]
                
                # The snippet uses:
                #  output_file_another = _path_ + r'/predicted_song'+'_'+str(index_check)+'.wav'
                # But if we always define index_check = len(all_notes_new), let's just do that:
                output_file_another = os.path.join(_path_, f"predicted_song_{index_check}.wav")
                output_file_another_direct = os.path.join(_path_, f"direct_from_mfcc_song_{index_check}.wav")
                output_file_input_samples = os.path.join(_path_, f"input_buffer_mfcc_samples_song_{index_check}.wav")
        
            # Step 3: Normalize the user input
            normalized_user_input_time_step = np.array(user_input_time_step.copy())
        
            # Reshape the input to match the model’s expected input format
            normalized_user_input_time_step_reshaped = normalized_user_input_time_step.reshape(
                (1, past_information_samples, characteristics_buff)
            )
        
            # Step 4: Pass the input data through the trained model to predict the next 180 time steps
            predictNextSequence_new = []
            
            for i in range(number_of_characteristics):
                if train_using_AdvancedTCN_faster:
                    test_tensor_new = torch.tensor(normalized_user_input_time_step_reshaped, dtype=torch.float32).to(device)
                    model = loaded_models[i].to(device)
                    model.eval()
                    with torch.no_grad():
                        output = model(test_tensor_new)
                    output = output.cpu().numpy()
                    temp_new = np.squeeze(output)[:future_information_samples]  # next 180
                    predictNextSequence_new.append(temp_new)
                else:
                    # If you have Keras models:
                    temp_new = loaded_models[i].predict(normalized_user_input_time_step_reshaped, verbose=1)
                    temp_new = temp_new[:future_information_samples]
                    predictNextSequence_new.append(temp_new)
        
            # Convert predictNextSequence_new into a final predicted array
            final_predicted_new_numpy = np.array(predictNextSequence_new).T
        
            # Step 5: Denormalize the predicted MFCCs (if applicable)
            if min_max_normalization:
                for i in range(number_of_characteristics):
                    final_predicted_new_numpy[:, i] = denormalize(
                        final_predicted_new_numpy[:, i],
                        characteristics_min[i],
                        characteristics_max[i]
                    )
            
            if z_score_normalization:
                for i in range(number_of_characteristics):
                    value_new = final_predicted_new_numpy[:, i]
                    param1, param2 = characteristics_params[i]
                    final_predicted_new_numpy[:, i] = z_score_denormalize(value_new, param1, param2)
        
            # Step 6: Handle non-finite values in the predicted MFCCs
            final_predicted_new_numpy = np.nan_to_num(final_predicted_new_numpy, nan=0.0, posinf=1.0, neginf=-1.0).T
        
            # Step 7: Convert the predicted MFCC matrix to an audio waveform and save it
            if encoding_method == "mfcc":
                mel_spec_new = mfcc_to_mel_spectrogram(final_predicted_new_numpy.T)
                mel_spec_new = np.nan_to_num(mel_spec_new, nan=0.0, posinf=1.0, neginf=-1.0)
                
                waveform_predicted_new = mel_to_waveform(mel_spec_new, new_sampling_rate)
                waveform_predicted_final_new = librosa.resample(waveform_predicted_new,
                                                                orig_sr=new_sampling_rate,
                                                                target_sr=sr_new)
                
                plt.figure()
                plt.plot(waveform_predicted_final_new)
                plt.show()
        
            # If user_defined_time_prediction_index == True, also save the direct-from-MFCC portion
            if user_defined_time_prediction_index:
                # final_matrix_actual_new
                final_matrix_actual_new = mfcc_new[:, index_check + past_information_samples : index_check + past_information_samples + future_information_samples]
                mel_spec_actual_new = mfcc_to_mel_spectrogram(final_matrix_actual_new.T)
                mel_spec_actual_new = np.nan_to_num(mel_spec_actual_new, nan=0.0, posinf=1.0, neginf=-1.0)
                
                waveform_mfcc_based_new = mel_to_waveform(mel_spec_actual_new, new_sampling_rate)
                waveform_mfcc_final_new = librosa.resample(waveform_mfcc_based_new, orig_sr=new_sampling_rate, target_sr=sr_new)
                
                plt.figure()
                plt.plot(waveform_mfcc_final_new)
                plt.show()
        
        
            sf.write(output_file_another_direct, waveform_mfcc_final_new, sr_new)

        
            # Step 8: Denormalize and save the waveform corresponding to the 600 input samples
            if min_max_normalization:
                for i in range(number_of_characteristics):
                    user_input_time_step[:, i] = denormalize(
                        user_input_time_step[:, i],
                        characteristics_min[i],
                        characteristics_max[i]
                    )
            
            if z_score_normalization:
                for i in range(number_of_characteristics):
                    param1, param2 = characteristics_params[i]
                    user_input_time_step[:, i] = z_score_denormalize(user_input_time_step[:, i], param1, param2)
        
            mel_spec_input = mfcc_to_mel_spectrogram(user_input_time_step)
            mel_spec_input = np.nan_to_num(mel_spec_input, nan=0.0, posinf=1.0, neginf=-1.0)
        
            waveform_input_samples = mel_to_waveform(mel_spec_input, new_sampling_rate)
            waveform_input_samples_resampled = librosa.resample(waveform_input_samples, orig_sr=new_sampling_rate, target_sr=sr_new)
            sf.write(output_file_input_samples, 0.95*waveform_input_samples_resampled, sr_new)
        
        
            
            
            filter_mfcc_approach = False
            if filter_mfcc_approach == True:    
        
                    # Example: Assume mfcc_nice and mfcc_bad are 2D numpy arrays of shape (num_coeffs, num_frames)
                    # For demonstration, we'll create random data. Replace these lines with your actual MFCC data.
                    np.random.seed(0)
                    mfcc_bad = final_predicted_new_numpy.copy()     # e.g., "unhealthy/jittery"
                    
                    # Compute statistics for both
                    def compute_statistics(mfcc_matrix):
                        """
                        Compute mean and standard deviation across both axes.
                        You can choose how you'd like to aggregate. Here, 
                        we'll compute per-coefficient stats and overall stats.
                        """
                        mean_per_coeff = np.mean(mfcc_matrix, axis=1)   # mean of each MFCC coefficient across frames
                        std_per_coeff = np.std(mfcc_matrix, axis=1)
                    
                        overall_mean = np.mean(mfcc_matrix)
                        overall_std = np.std(mfcc_matrix)
                    
                        return mean_per_coeff, std_per_coeff, overall_mean, overall_std
                    
                    mean_bad, std_bad, overall_mean_bad, overall_std_bad = compute_statistics(mfcc_bad)
                    
        
                    
                    print("/n=== BAD/JITTERY MFCC STATS ===")
                    print("Per-coeff mean:", mean_bad)
                    print("Per-coeff std:", std_bad)
                    print("Overall mean:", overall_mean_bad)
                    print("Overall std:", overall_std_bad)
                    
                    ##################################
                    # SMOOTHING THE BAD MFCCs
                    ##################################
                    
                    # Approach 1: Simple Moving Average along the time axis
                    def moving_average_filter(mfcc_matrix, window_size=5):
                        # Apply a moving average for each MFCC coefficient across frames
                        smoothed = np.copy(mfcc_matrix)
                        num_coeffs, num_frames = mfcc_matrix.shape
                        half_win = window_size // 2
                        for c in range(num_coeffs):
                            for t in range(num_frames):
                                start = max(0, t - half_win)
                                end = min(num_frames, t + half_win + 1)
                                smoothed[c, t] = np.mean(mfcc_matrix[c, start:end])
                        return smoothed
                    
                    mfcc_bad_ma = moving_average_filter(mfcc_bad, window_size=5)
                    
                    # Approach 2: Savitzky–Golay Filter (more sophisticated smoothing)
                    # This tries to fit local polynomials and can preserve some local structure.
                    # You'll need to choose parameters (window_length and polyorder) carefully.
                    window_length = 7  # must be odd and >= polyorder
                    polyorder = 2
                    mfcc_bad_sg = np.copy(mfcc_bad)
                    for c in range(mfcc_bad.shape[0]):
                        mfcc_bad_sg[c, :] = savgol_filter(mfcc_bad[c, :], window_length=window_length, polyorder=polyorder)
                    
                    # Now mfcc_bad_ma and mfcc_bad_sg contain smoothed versions of the "bad" MFCC array.
                    # You can re-check their stats or even attempt to invert back to audio.
                    
                    mean_bad_ma, std_bad_ma, overall_mean_bad_ma, overall_std_bad_ma = compute_statistics(mfcc_bad_ma)
                    mean_bad_sg, std_bad_sg, overall_mean_bad_sg, overall_std_bad_sg = compute_statistics(mfcc_bad_sg)
                    
                    print("/n=== BAD MFCC STATS AFTER MOVING AVERAGE SMOOTHING ===")
                    print("Overall mean:", overall_mean_bad_ma)
                    print("Overall std:", overall_std_bad_ma)
                    
                    print("/n=== BAD MFCC STATS AFTER SAVITZKY–GOLAY SMOOTHING ===")
                    print("Overall mean:", overall_mean_bad_sg)
                    print("Overall std:", overall_std_bad_sg)
                    
                    
                    
                    
                    output_file_another_direct2 = _path_ + r'/predicted_song_smoothed'+'_'+str(index_check)+'.wav'
                    
                    mel_spec_actual_new2 = mfcc_to_mel_spectrogram(mfcc_bad_ma.T)
                    
                    # Ensure Mel spectrogram is finite before converting
                    mel_spec_actual_new2 = np.nan_to_num(mel_spec_actual_new2, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Convert to waveform and save
                    waveform_mfcc_based_new2 = mel_to_waveform(mel_spec_actual_new2, new_sampling_rate)
                    waveform_mfcc_final_new2 = librosa.resample(waveform_mfcc_based_new2, orig_sr=new_sampling_rate, target_sr=sr_new)
                    sf.write(output_file_another_direct2, waveform_mfcc_final_new2, sr_new)
                    plt.figure()
                    plt.plot(waveform_mfcc_final_new2)
                    plt.show()
                    
                    
            
            
            
            filter_mel_approach1 = True
            if filter_mel_approach1 == True:    
                    
                    # Example: Assume mfcc_nice and mfcc_bad are 2D numpy arrays of shape (num_coeffs, num_frames)
                    # For demonstration, we'll create random data. Replace these lines with your actual MFCC data.
                    np.random.seed(0)
                    mfcc_bad = final_predicted_new_numpy.copy()     # e.g., "unhealthy/jittery"
                       
                    
                    def mel_to_stft(mel_spectrogram, sr=22050, n_fft=2048, n_mels=128, fmin=0.0, fmax=None):
                        """
                        Attempt to invert a mel-spectrogram back to an STFT magnitude spectrogram using a pseudo-inverse of the mel filter bank.
                    
                        Parameters
                        ----------
                        mel_spectrogram : np.ndarray [shape=(n_mels, t)]
                            Mel-spectrogram input.
                        sr : int
                            Sampling rate.
                        n_fft : int
                            Number of FFT bins used in the original STFT.
                        n_mels : int
                            Number of Mel bands used in the mel-spectrogram.
                        fmin : float
                            Minimum frequency of mel filter bank.
                        fmax : float or None
                            Maximum frequency of mel filter bank. If None, fmax = sr/2.
                    
                        Returns
                        -------
                        stft_magnitude : np.ndarray [shape=(1 + n_fft/2, t)]
                            Approximate magnitude spectrogram reconstructed from the mel-spectrogram.
                        """
                        if fmax is None:
                            fmax = sr / 2
                    
                        # Create the mel filter bank
                        mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
                    
                        # Pseudo-inverse of the mel filter bank
                        # mel_spectrogram = mel_filter * stft_magnitude
                        # => stft_magnitude ≈ mel_filter_pinv * mel_spectrogram
                        mel_filter_pinv = np.linalg.pinv(mel_filter)
                    
                        # Apply the pseudo-inverse
                        stft_magnitude = np.dot(mel_filter_pinv, mel_spectrogram)
                    
                        # Ensure no negative values due to numerical issues
                        stft_magnitude = np.maximum(0.0, stft_magnitude)
                    
                        return stft_magnitude
                    
                    # Example usage:
                    # mel_spec = ... # your mel spectrogram (n_mels x time)
                    # stft_mag_approx = mel_to_stft(mel_spec, sr=22050, n_fft=2048, n_mels=128, fmin=0, fmax=8000)
                    # Note that you only get magnitude. You can use something like Griffin-Lim to attempt phase reconstruction:
                    # waveform = librosa.griffinlim(stft_mag_approx)
                           
                            
                    
        
                    # Convert MFCC -> Mel-spectrogram -> STFT (Optional, if you prefer STFT domain)
                    mel_spec = mfcc_to_mel_spectrogram(mfcc_bad.T)
                    # Convert mel to linear-frequency spectrogram if you have the filter bank / inverse mel mapping
                    # This step depends on your pipeline and what data you have.
                    # Let's assume you have a function mel_to_stft(mel_spec) or known mel basis.
                    # If not, consider applying Wiener on mel_spec directly (although not as common).
                    
                    stft_matrix = mel_to_stft(mel_spec, sr=22050, n_fft=2048, n_mels=128, fmin=0, fmax=8000)  # You need to implement or have this function
                    magnitude, phase = librosa.magphase(stft_matrix)
                    
                    # Apply Wiener filter on the magnitude
                    magnitude_smoothed = sig.wiener(magnitude, mysize=(3,3))  # adjust window size
                    
                    # Recombine magnitude with original phase
                    stft_smoothed = magnitude_smoothed * phase
                    
                    # Invert STFT to time-domain
                    waveform_wiener = librosa.istft(stft_smoothed, hop_length=512, win_length=1024, window='hann')
                    
                    output_file_predicted_smoothed_wiener=_path_ + r'/predicted_song_wiener'+'_'+str(index_check)+'.wav'
                    sf.write(output_file_predicted_smoothed_wiener, 2.0*waveform_wiener, sr_new)
            
        
            # Also output input from user
            sf.write(output_file_input_initial, y_resampled_new, new_sampling_rate)
            
        
        
            # Collect the files we created
            generated_files.append(output_file_input_initial)
            generated_files.append(output_file_input_samples)
            generated_files.append(output_file_another)         
            generated_files.append(output_file_predicted_smoothed_wiener)
            if user_defined_time_prediction_index:
                generated_files.append(output_file_another_direct)
        
            # EXACT SNIPPET END
            # ----------------------------------------------------------------------
        
            return generated_files
        



    if encoding_method == "cqt":
            print("ok1")
            # Compute the Constant-Q Transform
            cqt_new = librosa.cqt(y_resampled_new, sr=new_sampling_rate, hop_length=hop_length, n_bins=n_freq_bins )# default n_bins=84, 56 is good enough

    
            # Split into real and imaginary parts
            cqt_real_new = cqt_new.real.T  # Shape: (Time Steps, Freq Bins)
            cqt_imag_new = cqt_new.imag.T  # Shape: (Time Steps, Freq Bins)
            
            # Concatenate along the feature dimension
            all_notes_new = np.concatenate([cqt_real_new, cqt_imag_new], axis=1)  # Shape: (Time Steps, 2 * Freq Bins)

            # Enlarge the signal for training, if necessary
            all_notes_new = np.tile(all_notes_new, (num_duplicates, 1))
            
            # Convert to float
            all_notes_new = all_notes_new.astype(float)
            
            if normalization_based_on_training_songs == True:
                if min_max_normalization:
                    characteristics_max,characteristics_min = run_normalization()
                    for i in range(len(all_notes_new[0])):
                        all_notes_new[:, i], _, _ = normalize(all_notes_new[:, i], 0, 1)
                
                if z_score_normalization:
                    characteristics_params = run_normalization()
                    for i in range(len(all_notes_new[0])):
                        all_notes_new[:, i], _, _ = z_score_normalize(all_notes_new[:, i])
     
                    
            if normalization_based_on_training_songs == False:        
                if min_max_normalization:                   
                    characteristics_max = []
                    characteristics_min = []
                    
                    for i in range(len(all_notes_new[0])):
                        range_to_normalize = (0, 1)
                        all_notes_new[:, i], characteristic_max, characteristic_min = normalize(all_notes_new[:, i],
                                                                                            range_to_normalize[0],
                                                                                            range_to_normalize[1])
                        characteristics_max.append(characteristic_max)
                        characteristics_min.append(characteristic_min)                   
                                    
                        
    
                if z_score_normalization:
                    characteristics_params = []
                    for i in range(len(all_notes_new[0])):
                        #all_notes_new[:, i], _, _ = z_score_normalize(all_notes_new[:, i])
                        all_notes_new[:, i], param1, param2  = z_score_normalize(all_notes_new[:, i])
                        characteristics_params.append((param1, param2))
                
                       
            
            
            
            print("ok2")
            # user_defined_time_prediction_index logic
            if user_defined_time_prediction_index and len(all_notes_new) > index_check + past_information_samples:
                # Take 600 time steps from index_check
                user_input_time_step = all_notes_new[index_check:index_check + past_information_samples, :]
                
                # We replicate exactly how the snippet constructs these paths
                # output_file_another, output_file_another_direct, output_file_input_samples
                output_file_another = os.path.join(_path_, f"predicted_song_{index_check}.wav")
                output_file_another_direct = os.path.join(_path_, f"direct_from_cqt_song_{index_check}.wav")
                output_file_input_samples = os.path.join(_path_, f"input_buffer_cqt_samples_song_{index_check}.wav")
                
            else:
                # user_defined_time_prediction_index = False
                # or data is not large enough
                index_check = len(all_notes_new)
                user_input_time_step = all_notes_new[-past_information_samples:, :]
                
                # The snippet uses:
                #  output_file_another = _path_ + r'/predicted_song'+'_'+str(index_check)+'.wav'
                # But if we always define index_check = len(all_notes_new), let's just do that:
                output_file_another = os.path.join(_path_, f"predicted_song_{index_check}.wav")
                output_file_another_direct = os.path.join(_path_, f"direct_from_cqt_song_{index_check}.wav")
                output_file_input_samples = os.path.join(_path_, f"input_buffer_cqt_samples_song_{index_check}.wav")
        
        
            print("ok3")
            # Step 3: Normalize the user input
            normalized_user_input_time_step = np.array(user_input_time_step.copy())
        
            # Reshape the input to match the model’s expected input format
            normalized_user_input_time_step_reshaped = normalized_user_input_time_step.reshape(
                (1, past_information_samples, characteristics_buff)
            )
        
            # Step 4: Pass the input data through the trained model to predict the next 180 time steps
            predictNextSequence_new = []
            
            for i in range(number_of_characteristics):
                if train_using_AdvancedTCN_faster:
                    test_tensor_new = torch.tensor(normalized_user_input_time_step_reshaped, dtype=torch.float32).to(device)
                    model = loaded_models[i].to(device)
                    model.eval()
                    with torch.no_grad():
                        output = model(test_tensor_new)
                    output = output.cpu().numpy()
                    temp_new = np.squeeze(output)[:future_information_samples]  # next 180
                    predictNextSequence_new.append(temp_new)
                else:
                    # If you have Keras models:
                    temp_new = loaded_models[i].predict(normalized_user_input_time_step_reshaped, verbose=1)
                    temp_new = temp_new[:future_information_samples]
                    predictNextSequence_new.append(temp_new)
        
            # Convert predictNextSequence_new into a final predicted array
            final_predicted_new_numpy = np.array(predictNextSequence_new).T
        
            # Step 5: Denormalize the predicted MFCCs (if applicable)
            if min_max_normalization:
                for i in range(number_of_characteristics):
                    final_predicted_new_numpy[:, i] = denormalize(
                        final_predicted_new_numpy[:, i],
                        characteristics_min[i],
                        characteristics_max[i]
                    )
            
            if z_score_normalization:
                for i in range(number_of_characteristics):
                    value_new = final_predicted_new_numpy[:, i]
                    param1, param2 = characteristics_params[i]
                    final_predicted_new_numpy[:, i] = z_score_denormalize(value_new, param1, param2)
            print("ok5")
            # Step 6: Handle non-finite values in the predicted MFCCs
            final_predicted_new_numpy = np.nan_to_num(final_predicted_new_numpy, nan=0.0, posinf=1.0, neginf=-1.0).T
        
            # Step 7: Convert the predicted CQT matrix to an audio waveform and save it
            
            # Split the predicted CQT array back into real and imaginary parts
            final_predicted_new_numpy = final_predicted_new_numpy.T
            real_part_predicted = final_predicted_new_numpy[:, :n_freq_bins]  # Shape: (Time Steps, Freq Bins)
            imag_part_predicted = final_predicted_new_numpy[:, n_freq_bins:]  # Shape: (Time Steps, Freq Bins)
            
            real_part_predicted = real_part_predicted.T  # Shape: (Freq Bins, Time Steps)
            imag_part_predicted = imag_part_predicted.T  # Shape: (Freq Bins, Time Steps)
                   
            cqt_reconstructed_predicted = real_part_predicted + 1j * imag_part_predicted  # Shape: (Freq Bins, Time Steps)
            
            print("ok6")
            if encoding_method == "cqt":
                # Inverse CQT (proper reconstruction)
                y_recon_predicted = librosa.icqt(
                    C=cqt_reconstructed_predicted,
                    sr=new_sampling_rate,
                    hop_length=hop_length
                )
                

                waveform_predicted_final_new = librosa.resample(y_recon_predicted,
                                                                orig_sr=new_sampling_rate,
                                                                target_sr=sr_new)
                
            
                
                #sf.write(output_file_another, waveform_predicted_final_new, sr_new)
                
                 # Use length_percentage to cut the signal
                print("length percentage: ",length_percentage)
                print(len(waveform_predicted_final_new))
                final_length = int(len(waveform_predicted_final_new) * length_percentage)
                waveform_cut = waveform_predicted_final_new[:final_length]
                print(len(waveform_cut))
                
                # Apply fade-in and fade-out smoothing to avoid clicks
                fade_duration_sec = 0.05  # 50 milliseconds fade duration
                fade_samples = int(fade_duration_sec * sr_new)
                # Ensure the fade duration is not longer than half of the signal
                fade_samples = min(fade_samples, len(waveform_cut) // 2)
            
                # Create fade-in and fade-out envelopes
                fade_in = np.linspace(0.0, 1.0, fade_samples)
                fade_out = np.linspace(1.0, 0.0, fade_samples)
                envelope = np.ones_like(waveform_cut)
                envelope[:fade_samples] = fade_in
                envelope[-fade_samples:] = fade_out
            
                # Apply the envelope to the cut waveform
                waveform_predicted_final_new_smoothed_and_cutted = waveform_cut * envelope
                           
                sf.write(output_file_another, waveform_predicted_final_new_smoothed_and_cutted, sr_new)                           
                
                # plt.figure()
                # plt.plot(waveform_predicted_final_new)
                # plt.title("predicted")
                # plt.show()
                


            # Step 8: Denormalize and save the waveform corresponding to the 600 input samples
            if min_max_normalization:
                for i in range(number_of_characteristics):
                    user_input_time_step[:, i] = denormalize(
                        user_input_time_step[:, i],
                        characteristics_min[i],
                        characteristics_max[i]
                    )
            
            if z_score_normalization:
                for i in range(number_of_characteristics):
                    param1, param2 = characteristics_params[i]
                    user_input_time_step[:, i] = z_score_denormalize(user_input_time_step[:, i], param1, param2)
        
  
            # Split the predicted CQT array back into real and imaginary parts
            real_part_user_input_time_step = user_input_time_step[:, :n_freq_bins]  # Shape: (Time Steps, Freq Bins)
            imag_part_user_input_time_step = user_input_time_step[:, n_freq_bins:]  # Shape: (Time Steps, Freq Bins)
            
            real_part_user_input_time_step = real_part_user_input_time_step.T  # Shape: (Freq Bins, Time Steps)
            imag_part_user_input_time_step = imag_part_user_input_time_step.T  # Shape: (Freq Bins, Time Steps)
                   
            cqt_reconstructed_user_input_time_step= real_part_user_input_time_step + 1j * imag_part_user_input_time_step  # Shape: (Freq Bins, Time Steps)
            
  
    
            waveform_input_samples = librosa.icqt(
                C=cqt_reconstructed_user_input_time_step,
                sr=new_sampling_rate,
                hop_length=hop_length
            )
            
            waveform_input_samples_resampled = librosa.resample(waveform_input_samples,
                                                       orig_sr=new_sampling_rate,
                                                       target_sr=sr_new)              
                            
           
            
            
            sf.write(output_file_input_samples, 0.95*waveform_input_samples_resampled, sr_new)
     
        



            ################################################################

            from scipy.ndimage import uniform_filter1d
            
            def enhance_variation(signal, window_size=1024, alpha=1.0):
                """
                Enhance the 'irregular' or 'varying' portion of an audio signal by
                unsharp masking: emphasizing differences from the local average.
            
                Parameters
                ----------
                signal : 1D numpy array
                    The original audio signal.
                window_size : int
                    Size of the smoothing window in samples. Larger = smoother “background.”
                alpha : float
                    Strength of the enhancement. 0 = no enhancement,
                    1.0 = typical unsharp effect, >1 for stronger effect.
            
                Returns
                -------
                enhanced_signal : 1D numpy array
                    The signal with amplified local variations.
                """
            
                # 1. Smooth the signal to get a "background"
                smoothed = uniform_filter1d(signal, size=window_size)
            
                # 2. Compute the difference (the 'detail' or 'irregular' part)
                difference = signal - smoothed
            
                # 3. Add scaled difference back to original
                enhanced_signal = signal + alpha * difference
            
                return enhanced_signal
            
            # Example usage:
            # signal is your 1D numpy array of audio samples
            # window_size might be ~1-5 ms for transient emphasis or bigger if the melody changes more slowly
            # alpha controls how much we boost the variation
            # e.g., 1.0 is a good starting point; try 0.5 or 2.0 to taste.

            
            from scipy.signal import butter, sosfiltfilt
            
            def eq_melody_boost(signal, sample_rate, lowcut, highcut, order=5, gain_db=6):
                """            output_file_another1 = os.path.join(_path_, f"predicted_song_{index_check}_1.wav")

                Boost frequency bands containing melody using a Butterworth bandpass filter.
                
                Parameters:
                    signal (np.array): Input audio signal
                    sample_rate (int): Sampling rate (Hz)
                    lowcut (float): Lower frequency bound (Hz)
                    highcut (float): Upper frequency bound (Hz)
                    order (int): Filter order (higher = steeper rolloff)
                    gain_db (float): Boost amount in decibels
                
                Returns:
                    np.array: EQ-enhanced signal
                """
                # Convert gain from dB to linear scale
                gain = 10 ** (gain_db / 20)
                
                # Design bandpass filter
                nyq = 0.5 * sample_rate
                sos = butter(order, [lowcut/nyq, highcut/nyq], btype='band', output='sos')
                
                # Apply filter (zero-phase)
                filtered = sosfiltfilt(sos, signal)
                
                # Mix original and filtered signal
                return signal + (filtered * gain)
            
            # Example: Boost 200-2000 Hz range for vocal melody
            # enhanced = eq_melody_boost(signal, 44100, 200, 2000, gain_db=9)
            
        
            
            def isolate_harmonic(signal, sample_rate, margin=3.0, power=5.0):
                """            output_file_another1 = os.path.join(_path_, f"predicted_song_{index_check}_1.wav")

                Separate harmonic (melodic) components using HPS.
                
                Parameters:
                    signal (np.array): Input audio signal
                    sample_rate (int): Sampling rate
                    margin (float): Separation strength (higher = more separation)
                    power (float): Spectral difference exponent
                
                Returns:
                    np.array: Isolated harmonic component
                """
                # Compute STFT
                D = librosa.stft(signal)
                
                # Separate components
                H, P = librosa.decompose.hpss(D, margin=margin, kernel_size=31, power=power)
                
                # Reconstruct harmonic signal
                return librosa.istft(H)
            
            # Example usage:
            # harmonic = isolate_harmonic(signal, 44100, margin=5.0)
            
            
            def dynamic_compressor(signal, sample_rate, threshold_db=-20.0, ratio=4.0, attack=0.01, release=0.1):
                """
                Compress dynamics to increase sustain of melodic notes.
                
                Parameters:
                    signal (np.array): Input audio signal
                    threshold_db (float): Compression threshold (dB)
                    ratio (float): Compression ratio (4:1, etc.)
                    attack (float): Attack time (seconds)
                    release (float): Release time (seconds)
                
                Returns:
                    np.array: Compressed signal
                """
                # Convert dB threshold to linear amplitude
                threshold = 10 ** (threshold_db / 20)
                
                # Initialize envelope and gain
                envelope = np.abs(signal)
                gain_reduction = np.ones_like(signal)
                
                # Smoothing filter coefficients
                alpha_attack = np.exp(-1/(attack * sample_rate))
                alpha_release = np.exp(-1/(release * sample_rate))
                
                # Compute gain reduction
                for i in range(1, len(signal)):
                    if envelope[i] > threshold:
                        alpha = alpha_attack
                    else:
                        alpha = alpha_release
                        
                    gain_reduction[i] = alpha * gain_reduction[i-1] + (1-alpha) * (
                        1 - (1/ratio) * (envelope[i] > threshold)
                    )
                
                # Apply gain
                compressed = signal * gain_reduction
                
                # Normalize to prevent clipping
                return compressed / np.max(np.abs(compressed))
            
            # Example: Gentle compression for piano sustain
            # compressed = dynamic_compressor(signal, sr_new, threshold_db=-15, ratio=3.0)
            


            ################################################################

            reference_rms = np.sqrt(np.mean(waveform_input_samples_resampled**2))


            def match_rms_level(signal, target_rms, eps=1e-9):
                """
                Scale 'signal' so that its RMS matches 'target_rms'.
                To avoid division by zero, we use 'eps' as a small offset.
                """
                # Measure RMS of the signal
                current_rms = np.sqrt(np.mean(signal**2)) + eps
                
                # Compute scaling factor
                gain = target_rms / current_rms
                
                # Apply gain
                return signal * gain
            


            # enhanced_signal1 = enhance_variation(waveform_predicted_final_new_smoothed_and_cutted, window_size=4092, alpha=2.0)
            # enhanced_signal1 = match_rms_level(enhanced_signal1, reference_rms)
            # output_file_another1 = os.path.join(_path_, f"predicted_song_{index_check}_1.wav")
            # sf.write(output_file_another1, enhanced_signal1, sr_new)
            # enhanced_signal2 = enhance_variation(waveform_predicted_final_new_smoothed_and_cutted, window_size=16000, alpha=2.0)
            # enhanced_signal2= match_rms_level(enhanced_signal2, reference_rms)
            # output_file_another2 = os.path.join(_path_, f"predicted_song_{index_check}_2.wav")
            # sf.write(output_file_another2, enhanced_signal2, sr_new)
            # enhanced_signal3 = dynamic_compressor(waveform_predicted_final_new_smoothed_and_cutted, sr_new, threshold_db=-15, ratio=3.0)
            # enhanced_signal3 = match_rms_level(enhanced_signal3, reference_rms)
            # output_file_another3 = os.path.join(_path_, f"predicted_song_{index_check}_3.wav")
            # sf.write(output_file_another3, enhanced_signal3, sr_new)
            # enhanced_signal4 = 5.0*isolate_harmonic(waveform_predicted_final_new_smoothed_and_cutted,  sr_new, margin=1.0)
            # enhanced_signal4 = match_rms_level(enhanced_signal4, reference_rms)
            # output_file_another4 = os.path.join(_path_, f"predicted_song_{index_check}_4.wav")
            # sf.write(output_file_another4, enhanced_signal4, sr_new)
            enhanced_signal4a = 5.0*isolate_harmonic(waveform_predicted_final_new_smoothed_and_cutted,  sr_new, margin=5.0)
            enhanced_signal4a = match_rms_level(enhanced_signal4a, reference_rms)
            output_file_another4a = os.path.join(_path_, f"predicted_song_{index_check}_1a.wav")
            sf.write(output_file_another4a, enhanced_signal4a, sr_new)
            enhanced_signal4b = 5.0*isolate_harmonic(waveform_predicted_final_new_smoothed_and_cutted,  sr_new, margin=12.0)
            enhanced_signal4b = match_rms_level(enhanced_signal4b, reference_rms)
            output_file_another4b = os.path.join(_path_, f"predicted_song_{index_check}_1b.wav")
            sf.write(output_file_another4b, enhanced_signal4b, sr_new)
            enhanced_signal5 = eq_melody_boost(waveform_predicted_final_new_smoothed_and_cutted, sr_new, 200, 2000, gain_db=9)
            enhanced_signal5 = match_rms_level(enhanced_signal5, reference_rms)
            output_file_another5 = os.path.join(_path_, f"predicted_song_{index_check}_2.wav")
            sf.write(output_file_another5, enhanced_signal5, sr_new)
            # enhanced_signal6 = enhance_variation(waveform_predicted_final_new_smoothed_and_cutted, window_size=4092, alpha=1.0)
            # enhanced_signal6 = match_rms_level(enhanced_signal6, reference_rms)
            # output_file_another6 = os.path.join(_path_, f"predicted_song_{index_check}_6.wav")
            # sf.write(output_file_another6, enhanced_signal6, sr_new)            
            
                
            plt.figure()
            plt.plot(waveform_predicted_final_new_smoothed_and_cutted)
            plt.title("predicted")
            plt.show()
            
            # plt.figure()
            # plt.plot(enhanced_signal1)
            # plt.title("predicted enhanced_signal")
            # plt.show()            
            # plt.figure()
            # plt.plot(enhanced_signal2)
            # plt.title("predicted enhanced_signal")
            # plt.show() 
            # plt.figure()
            # plt.plot(enhanced_signal3)
            # plt.title("predicted enhanced_signal")
            # plt.show() 
            # plt.figure()
            # plt.plot(enhanced_signal4)
            # plt.title("predicted enhanced_signal")
            # plt.show() 
            plt.figure()
            plt.plot(enhanced_signal4a)
            plt.title("predicted enhanced_signal")
            plt.show() 
            plt.figure()
            plt.plot(enhanced_signal4b)
            plt.title("predicted enhanced_signal")
            plt.show() 
            plt.figure()
            plt.plot(enhanced_signal5)
            plt.title("predicted enhanced_signal")
            plt.show() 
            # plt.figure()
            # plt.plot(enhanced_signal6)
            # plt.title("predicted enhanced_signal")
            # plt.show()                 










            print("ok7")
            # If user_defined_time_prediction_index == True, also save the direct-from-CQT portion
            if user_defined_time_prediction_index:
                # final_matrix_actual_new
                final_matrix_actual_new = cqt_new[:, index_check + past_information_samples : index_check + past_information_samples + future_information_samples]

                waveform_cqt_final_new = librosa.icqt(
                    C=final_matrix_actual_new,
                    sr=new_sampling_rate,
                    hop_length=hop_length
                )
                
                waveform_cqt_final_new2 = librosa.resample(waveform_cqt_final_new,
                                                                orig_sr=new_sampling_rate,
                                                                target_sr=sr_new)              
                
                sf.write(output_file_another_direct, waveform_cqt_final_new2, sr_new)
                
                plt.figure()
                plt.plot(waveform_cqt_final_new2)
                plt.show()
        
        
        
            # Also output input from user
            sf.write(output_file_input_initial, y_resampled_new, new_sampling_rate)
            
        
        
            # Collect the files we created
            generated_files.append(output_file_input_initial)
            generated_files.append(output_file_input_samples)
            generated_files.append(output_file_another)
            # generated_files.append(output_file_another1)
            # generated_files.append(output_file_another2)
            # generated_files.append(output_file_another3)
            # generated_files.append(output_file_another4)
            generated_files.append(output_file_another4a)
            generated_files.append(output_file_another4b)
            generated_files.append(output_file_another5)
            # generated_files.append(output_file_another6)
            

        
            # EXACT SNIPPET END
            # ----------------------------------------------------------------------
        
            return generated_files
        
        
        
    
new_filles=run_inference(_path_, (audio_file_another), (user_defined_time_prediction_index), (index_check), float(length_percentage))




