# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:58:35 2024

@author: alberto
"""


print("hi")

#!/usr/bin/env python3
import argparse
import sys
import os 

# ─── Fix Matplotlib for headless / batch mode ────────────────────
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


import argparse

parser = argparse.ArgumentParser(
    description="Load a MIDI, run TCN generator, write results"
)
parser.add_argument(
    "--input", "-i",
    required=True,
    help="Path to the input MIDI file"
)
parser.add_argument(
    "--output", "-o",
    required=True,
    help="Directory where generated files will be saved"
)
parser.add_argument(
    "--mode", "-m",
    choices=["option_1", "option_2"],
    default="option_1",
    help="Which processing mode to use"
)
parser.add_argument(
    "--skip_notes_lower_than", "-s",
    type=int,
    default=0,
    help="Ignore all notes with MIDI pitch below this threshold"
)
parser.add_argument(
    "-r", "--retrain",
    action="store_true",
    help="If set, retrain the model after generating the output."
)

args = parser.parse_args()

input_path            = args.input
output_dir            = args.output
mode                  = args.mode
skip_notes_lower_than = args.skip_notes_lower_than
retrain               = args.retrain

print(f"→ Input MIDI   : {input_path}")
print(f"→ Output Dir   : {output_dir}")
print(f"→ Mode         : {mode}")
print(f"→ Skip < pitch : {skip_notes_lower_than}")
print(f"→ Retrain      : {retrain}")

    

from pathlib import Path

# this will give you the folder containing this script:
SCRIPT_DIR = Path(__file__).resolve().parent

# if you want the project root (one level up):
PROJECT_ROOT = SCRIPT_DIR.parent.parent# this is v0.01


_path_ = SCRIPT_DIR

## Create models subfolder  1st AI
# checking if the directory exist or not. 
if not os.path.exists(_path_/"AImodel1"):      
    # if the directory is not present  then create it. 
    os.makedirs(_path_/"AImodel1") 

"""
#########################################################################################################
#########################################################################################################
Step A: Training phase
-> step A1 : Read any midi fille and convert it to suitable csv fille
#########################################################################################################
#########################################################################################################
"""

from mido import MidiFile
import numpy as np
import string   
   



mid = MidiFile(input_path, clip=True)
print(mid)

def MidiStringToInt(midstr):
    Notes = [["C"],["C#","Db"],["D"],["D#","Eb"],["E"],["F"],["F#","Gb"],["G"],["G#","Ab"],["A"],["A#","Bb"],["B"]]
    answer = 0
    i = 0
    #Note
    letter = midstr.split('-')[0].upper()
    for note in Notes:
        for form in note:
            if letter.upper() == form:
                answer = i
                break;
        i += 1
    #Octave
    answer += (int(midstr[-1]))*12
    return answer

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)
errors = {
    'program': 'Bad input, please refer this spec-\n'
               'http://www.electronics.dit.ie/staff/tscarff/Music_technology/midi/program_change.htm',
    'notes': 'Bad input, please refer this spec-\n'
             'http://www.electronics.dit.ie/staff/tscarff/Music_technology/midi/midi_note_numbers_for_octaves.htm'
}
def number_to_note(number: int) -> tuple:
    octave = number // NOTES_IN_OCTAVE
    assert octave in OCTAVES, errors['notes']
    assert 0 <= number <= 127, errors['notes']
    note = NOTES[number % NOTES_IN_OCTAVE]

    return note, octave

def msg2dict(msg):
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]


def switch_note(last_state, note, velocity, on_=True):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    result = [0] * 88 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note-21] = velocity if on_ else 0
    return result


def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]

def track2seq(track):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0]*88)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result += [last_state]*new_time
        last_state, last_time = new_state, new_time
    return result

def mid2arry(mid, min_msg_pct=0.1):
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i])
            all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]



result_array = mid2arry(mid)
plt.plot(range(result_array.shape[0]), np.multiply(np.where(result_array>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
plt.title("spring_no2_adagio_gp.mid")
plt.show()



"check:"
"piano has 88 notes, corresponding to note id 21 to 108"
note_offset_fix=21
time_index=1
audio_start=1
audio_end=500
for i in range(audio_start,audio_end):
    print("time index: ", i)
    for j in range(88):   
        if result_array[i,j]>0:
            print("note:",number_to_note(j+note_offset_fix),j+note_offset_fix,"velocity:",result_array[i,j])


np.savetxt('result_array.csv', result_array)


"""
Convert result_array to have one more coloumn that says how many times we have 
the same row (same note combinations) and delete dublicates - v2
"""
row_durations=[]
count = 0
for i in range(len(result_array)-1):
    if np.array_equal(result_array[i+1,:], result_array[i,:]) == True:
        count = count + 1
    else:
        count = count + 1
        row_durations.append(count)
        count = 0
    if i == len(result_array)-2:
        count = count + 1
        row_durations.append(count)
        count = 0        
        
new_array = np.zeros((len(row_durations),89)) 
duration_sum = 0       
for i in range(len(row_durations)):
    duration_sum = duration_sum + row_durations[i]
    new_array[i,0:88]=(result_array[duration_sum-1,:])
    new_array[i,88]=row_durations[i]
summary_notes_array=new_array.astype(int)




"""export notes to csv fille:"""
np.savetxt('summary_notes_reference_piece.csv', summary_notes_array)
# "to check reloading:"
datax = np.loadtxt('summary_notes_reference_piece.csv')
datax=datax.astype(int)   

    

"to check efficient reconstuction to initial result_array from summary_notes_array:"
#preallocate:
no_of_rows = np.sum(datax[:,88])    
reconstructed_result_array=np.zeros((no_of_rows,89))
temp=datax
indexed_sum=0
for i in range(len(datax[:,0])):
    new_element = temp[i,:].reshape(1,-1) 
    reconstructed_result_array[indexed_sum:indexed_sum+datax[i,88],:]=new_element 
    indexed_sum = indexed_sum + datax[i,88]
    
reconstructed_result_array = np.delete(reconstructed_result_array,88,1)  
reconstructed_result_array = reconstructed_result_array.astype(int)
"check for correct reconstruction:"
print("##########")
initial_array = mid2arry(mid)
print("correct reconstruction?:",np.array_equal(reconstructed_result_array, initial_array))
print("##########") 


" To recreate song in midi and check if it is the same (but reduced for piano)"
recreate_midi = True
if recreate_midi == True:
    import mido
    
    def arry2mid(ary, tempo=500000):
        # get the difference
        new_ary = np.concatenate([np.array([[0] * 88]), np.array(ary)], axis=0)
        changes = new_ary[1:] - new_ary[:-1]
        # create a midi file with an empty track
        mid_new = mido.MidiFile()
        track = mido.MidiTrack()
        mid_new.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        # add difference in the empty track
        last_time = 0
        for ch in changes:
            if set(ch) == {0}:  # no change
                last_time += 1
            else:
                on_notes = np.where(ch > 0)[0]
                on_notes_vol = ch[on_notes]
                off_notes = np.where(ch < 0)[0]
                first_ = True
                for n, v in zip(on_notes, on_notes_vol):
                    new_time = last_time if first_ else 0
                    track.append(mido.Message('note_on', note=n + 21, velocity=v, time=new_time))
                    first_ = False
                for n in off_notes:
                    new_time = last_time if first_ else 0
                    track.append(mido.Message('note_off', note=n + 21, velocity=0, time=new_time))
                    first_ = False
                last_time = 0
        return mid_new
    
    
    mid_new = arry2mid(reconstructed_result_array, 545455)
    mid_new.save(str(output_dir)+'/reference_piece_piano_version.mid')



"""
#########################################################################################################
#########################################################################################################
Step A: Training phase
-> step A2 : Train AI model 1 
#########################################################################################################
#########################################################################################################
"""

import numpy as np
import tensorflow as tf
print(tf. __version__)
import keras
print("Standalone Keras version:", keras.__version__)
import time
#tf.compat.v1.disable_eager_execution()


# # Patch get_custom_objects (if not already patched)
# keras.saving.get_custom_objects = tf.keras.utils.get_custom_objects

# # Patch register_keras_serializable to point to the correct function in tf.keras.utils
# keras.saving.register_keras_serializable = tf.keras.utils.register_keras_serializable

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
    
    def call(self, inputs):
        x = self.lstm(inputs)  
        #x = self.linear(x)   
        x = self.dense(x[:, -1, :])
        x = tf.reshape(x, (-1, output_predicted_chords,1))  # Reshape to (batch_size, 2, output_size)
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
        self.dense = tf.keras.layers.Dense(output_size, activation='linear')

    def call(self, inputs):
        x = inputs
        for layer in self.conv1d_layers:
            x = layer(x)
        x = self.dense(x[:, -1, :])
        return x

################################################################################
################################################################################
# Note inputs
#datax = numpy.loadtxt('result_array.csv')[1:10000]
datax = np.loadtxt('summary_notes_reference_piece.csv')
#all_notes = datax.astype(int)   

####################### Data Normalization ################################
# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr) 
    characteristic_max= max(arr)
    characteristic_min=min(arr)  
    if np.isnan(diff_arr) == False:
        for i in arr:
            temp = (((i - min(arr))*diff)/diff_arr) + t_min
            norm_arr.append(temp)
    if np.isnan(diff_arr) == True:
        for i in arr:
            norm_arr.append(0.0)     
    return norm_arr,characteristic_max,characteristic_min

all_notes = datax.copy()
characteristics_max=[]
characteristics_min=[]
#Normalize each feature indipendently:
for i in range(89):
    range_to_normalize = (0,1)
    all_notes[:,i],characteristic_max,characteristic_min = normalize(all_notes[:,i],
                                range_to_normalize[0],
                                range_to_normalize[1])
    characteristics_max.append(characteristic_max)
    characteristics_min.append(characteristic_min)
    
all_notes=np.array(all_notes)    
all_notes[np.isnan(all_notes)] = 0
######################################################################









input_chords = 60
output_predicted_chords = 30







######################################################################
##############  INPUT DATA
######################################################################
#This is the time buffer (how many time samples will be used):
buffer_length=input_chords
#This is the ammount of the characteristics choosen starting from the all_characteristics index:
characteristics_buff=89
#This is the len of the characteristics: 
#(choose 89 if duration should be also considered)
#(choose 88 if duration shouldn't be considered)
#(choose 41 to include characteristics between 41-characteristics_buff:41
all_characteristics=89
inputs=np.full((len(all_notes)-buffer_length-output_predicted_chords,buffer_length,characteristics_buff), 0.0)

# Separate to inputs-outputs
for time_index in range(len(all_notes)-buffer_length-output_predicted_chords): #number of changes
    input_temp=all_notes[time_index:time_index+buffer_length,all_characteristics-characteristics_buff:all_characteristics]
    inputs[time_index,0:buffer_length,0:characteristics_buff]=(input_temp)

# len(all_notes) samples, each with buffer_length time steps and 89 features
input_data1 = inputs

################################################################################
################################################################################ 

######################################################################
##############  OUTPUT DATA
######################################################################

output_data1=[]
num_samples = len(all_notes) - buffer_length-output_predicted_chords  # Adjust for the two future values
for i in range(89):
    #This is the time buffer (how many time samples will be used):
    buffer_length=input_chords
    outputs=np.full((len(all_notes)-buffer_length,1,1), 0.0)
    
    SPESIFIC_CHARACTERISTIC_TO_PREDICT = i
    
    outputs = np.full((num_samples, output_predicted_chords, 1), 0.0)
    for time_index in range(num_samples):
        output_temp = all_notes[time_index + buffer_length:time_index + buffer_length + output_predicted_chords, i]
        for k in range(output_predicted_chords):
            outputs[time_index, k, 0] = output_temp[k]
    output_data1.append(outputs)
################################################################################
################################################################################   


################################################################################
################################################################################   
################################################################################
################################################################################   
retrain=retrain
################################################################################
################################################################################   
################################################################################
################################################################################   

if retrain == True:
    model=[]
    for i in range(89):  #Default : 89. Use <89 to learn for less characteristics
        model.append(MultidimensionalLSTM(hidden_size=150,hidden_size2=328, output_size=output_predicted_chords))
        '''
        #://www.tensorflow.org/api_docs/python/tf/keras/losses
        '''
        #model.compile(optimizer="adam", loss="mse")
        model[i].compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
        #model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        
        print("################ Characteristic: ",i,"  ########################")
        # We train the model using:
        # output_data1[i] --> coresponds to a 1d list of all the values of the i characteristic
        # input_data11 --> coresponds to a 3d matrix 
        model[i].fit(input_data1, output_data1[i], epochs=40)
        print("################  end  ########################")
        print("###############################################")
        # path=str(_path_)+r"/AImodel1/"
        # model[i].save(path+"custom_model"+str(i)+".keras")
    
    
    #Save ML models
    for i in range(0,89):
       path=str(_path_)+r"/AImodel1/"
       model[i].save(path+"custom_model"+str(i)+".keras")

    #Load ML models
    loaded_models=[]
    for i in range(0,89):
       path=str(_path_)+r"/AImodel1/"
       loaded_models.append(keras.models.load_model(path+"custom_model"+str(i)+".keras"))
        
        
    #Check 
    # print("if following are equal means that loaded models are the same with the saved")
    # test_data = input_data[184,:,:]
    # test_data = test_data.reshape((1, buffer_length, characteristics_buff))   
    # print(model[1].predict(test_data, verbose=1))
    # print(loaded_models[1].predict(test_data, verbose=1))

if retrain == False:
    #Load ML models
    loaded_models=[]
    for i in range(0,89):
       path=str(_path_)+r"/AImodel1/"
       print("############")
       print(path+"custom_model"+str(i)+".keras")
       loaded_models.append(keras.models.load_model(path+"custom_model"+str(i)+".keras"))
        
            

"To check if the predictions are same with the initial song"
check_prediction_accuracy = False
if check_prediction_accuracy == True:
    t0 = time.time()
    final_predicted=[]
    for index in range(0,20): #for each state change (maximum value: (row_durations - buffer_length))
        predictNextSequence=[]
        real=[]
        test_data = input_data1[index,:,:]
        test_data = test_data.reshape((1, buffer_length, characteristics_buff))
        for i in range(0,89): #for each characteristic
            temp = loaded_models[i].predict(test_data, verbose=1)
            temp[temp<0.01] = 0
            predictNextSequence.append(temp)
            real.append(output_data1[i][index,:,:])
        # print('real:')
        # print(np.round(real,4))
        # print('predicted:')
        # print(np.round(predictNextSequence,4))
        # final_predicted.append(np.round(predictNextSequence,4))
        final_predicted.append(np.transpose(predictNextSequence, axes = (output_predicted_chords,1,0)))
        #print("#################################################################")
        print("index: ",index)
    t1 = time.time()
    print('prediction duration=',t1-t0)
    

    
    # ######################################################################
    ### SUGGESTIONS:
    ### 1. LINE 432 should have length 734 to predict the whole piece
    ### 2. LINE 394 change epochs to arround 50-130 for improved accuracy
    # ######################################################################
    
    
    # Restore final_predicted to have the same format with all_notes
    final_predicted_numpy = np.array(final_predicted)
    final_predicted_numpy = final_predicted_numpy[:,0,0,:]
    
    # Denormalize in order to reconstruct the piece
    ####################### Data De-Normalization ################################
    # explicit function to denormalize array
    def denormalize(arr, t_min, t_max):
        denorm_arr = []
        diff = 1
        diff_arr = t_max - t_min  
        if np.isnan(diff_arr) == False:
            for i in arr:
                temp = t_min+((i-0)*diff_arr/diff)
                denorm_arr.append(temp)           
        return denorm_arr
    
    final_predicted_numpy_actual = final_predicted_numpy.copy()
    #De-Normalize each feature indipendently:
    for i in range(89):
        final_predicted_numpy_actual[:,i] = denormalize(final_predicted_numpy[:,i],
                                    characteristics_min[i],
                                    characteristics_max[i])
        
    final_predicted_numpy_actual=np.array(final_predicted_numpy_actual)    
    final_predicted_numpy_actual[np.isnan(final_predicted_numpy_actual)] = 0
    final_predicted_numpy_actual[final_predicted_numpy_actual<4] = 0
    final_predicted_numpy_actual=np.round(final_predicted_numpy_actual)
    
    ######################################################################
    
    
    # """export predicted notes to csv fille:"""
    print("Exporting to csv. Note that the exported file has buffer_length less elements than the original")
    np.savetxt('final_predicted_AI_from_training.csv', final_predicted_numpy_actual)









"""
#########################################################################################################
#########################################################################################################
Step A: Training phase
-> step A5 : Convert Tensorflow models to tensorflow lite models for 100x faster prediction
#########################################################################################################
#########################################################################################################
"""
# references:
#https://micwurm.medium.com/using-tensorflow-lite-to-speed-up-predictions-a3954886eb98
#https://github.com/tensorflow/tensorflow/issues/53101

##############  Transform models to ltlite models for faster pprediction speed
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
        # out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        # for i in range(count):
        #     self.interpreter.set_tensor(self.input_index, inp[i:i+1])
        #     self.interpreter.invoke()
        #     out[i] = self.interpreter.get_tensor(self.output_index)[0]
        # return out
        # # Assuming self.output_shape is something like (2, 1), we need to handle it properly
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


### first we need to do some warm up by utilising the initial keras models
buffer_length1 = input_chords
if True:
    ################################################   example input 3 (training song's pattern)
    #initialize:
    user_input_time_step1 = []
    for i in range(buffer_length1):
        user_input_time_step1.append(np.zeros(89))
    # add some random notes (with hit power). 
    index_check = 45
    user_input_time_step1 =  input_data1[index_check,:,:]

        
##### Normalize the inserted values:
normalized_user_input_time_step1 = np.array(user_input_time_step1.copy())
normalized_user_input_time_step_reshaped1 = normalized_user_input_time_step1.reshape((1, buffer_length1, characteristics_buff))    

for i in range(89):
    temp = loaded_models[i].predict(normalized_user_input_time_step_reshaped1, verbose=0)


### Then we can convert to actual ltlite models
lmodels = []

for i in range(89):
    lmodels.append(LiteModel.from_keras_model(loaded_models[i]))



"""
#########################################################################################################
#########################################################################################################
Step B: Now that we have ready the model(s) we can create the suggestions for the user input
-> step B1 : Showcase the suggestion routine
#########################################################################################################
#########################################################################################################
"""
#example 1 is predictions on ranother piece
#example 2 is to check model's accuracy with given piece


import copy



def reconstruct_array(datax):
    no_of_rows = np.sum(datax[:,88])    
    reconstructed_result_array=np.zeros((no_of_rows,89))
    temp=datax
    indexed_sum=0
    for i in range(len(datax[:,0])):
        new_element = temp[i,:].reshape(1,-1) 
        reconstructed_result_array[indexed_sum:indexed_sum+datax[i,88],:]=new_element 
        indexed_sum = indexed_sum + datax[i,88]
        
    reconstructed_result_array = np.delete(reconstructed_result_array,88,1)  
    reconstructed_result_array = reconstructed_result_array.astype(int)
    return reconstructed_result_array





def arry2mid(ary, tempo=500000):
        # get the difference
        new_ary = np.concatenate([np.array([[0] * 88]), np.array(ary)], axis=0)
        changes = new_ary[1:] - new_ary[:-1]
        # create a midi file with an empty track
        mid_new = mido.MidiFile()
        track = mido.MidiTrack()
        mid_new.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        # add difference in the empty track
        last_time = 0
        for ch in changes:
            if set(ch) == {0}:  # no change
                last_time += 1
            else:
                on_notes = np.where(ch > 0)[0]
                on_notes_vol = ch[on_notes]
                off_notes = np.where(ch < 0)[0]
                first_ = True
                for n, v in zip(on_notes, on_notes_vol):
                    new_time = last_time if first_ else 0
                    track.append(mido.Message('note_on', note=n + 21, velocity=v, time=new_time))
                    first_ = False
                for n in off_notes:
                    new_time = last_time if first_ else 0
                    track.append(mido.Message('note_off', note=n + 21, velocity=0, time=new_time))
                    first_ = False
                last_time = 0
        return mid_new





import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pretty_midi
import soundfile as sf
#from playsound import playsound
#import wave
import os



###########################################################
# 1) Put your existing "example == 1" code into a function
###########################################################
def run_example_1(selected_midi_path,skip_notes_lower_than):
        global predictNextSequence
        global real_notes_new
        """
        Implement the entire logic from the 'if example == 1:' block,
        reading 'selected_midi_path' as the .mid file
        and eventually producing:
         - final_predicted_numpy_actual.mid
         - final_predicted_numpy_actual2.mid
         - final_predicted_numpy_actual3.mid
         - final_real_numpy_actual.mid
        or whichever files you create. 
        """
    
        mid_new = MidiFile(selected_midi_path, clip=True)
        print(mid_new)
        

        
        result_array_new = mid2arry(mid_new)
        import matplotlib.pyplot as plt
        plt.plot(range(result_array_new.shape[0]), np.multiply(np.where(result_array_new>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
        plt.title("vivaldi_concerto_la_stravaganza_4_2.mid")
        plt.show()
        

        
        """
        Convert result_array to have one more coloumn that says how many times we have 
        the same row (same note combinations) and delete dublicates - v2
        """
        row_durations=[]
        count = 0
        for i in range(len(result_array_new)-1):
            if np.array_equal(result_array_new[i+1,:], result_array_new[i,:]) == True:
                count = count + 1
            else:
                count = count + 1
                row_durations.append(count)
                count = 0
            if i == len(result_array_new)-2:
                count = count + 1
                row_durations.append(count)
                count = 0        
                
        new_array = np.zeros((len(row_durations),89)) 
        duration_sum = 0       
        for i in range(len(row_durations)):
            duration_sum = duration_sum + row_durations[i]
            new_array[i,0:88]=(result_array_new[duration_sum-1,:])
            new_array[i,88]=row_durations[i]
        summary_notes_array_new=new_array.astype(int)
        

        """export notes to csv fille:"""
        np.savetxt('summary_notes_another_piece.csv', summary_notes_array_new)
        # "to check reloading:"
        
        #datax = np.loadtxt('summary_notes_reference_piece.csv')
        datax_new = np.loadtxt('summary_notes_another_piece.csv')
          
        
     
    
        ####################### Data Normalization ################################

        all_notes_new = datax_new.copy()
        characteristics_max_new=[]
        characteristics_min_new=[]

        #Normalize each feature indipendently:
        for i in range(89):
            range_to_normalize = (0,1)
            all_notes_new[:,i],characteristic_max_new,characteristic_min_new = normalize(all_notes_new[:,i],
                                        range_to_normalize[0],
                                        range_to_normalize[1])
            characteristics_max_new.append(characteristic_max_new)
            characteristics_min_new.append(characteristic_min_new)

            
        all_notes_new=np.array(all_notes_new)    
        all_notes_new[np.isnan(all_notes_new)] = 0
        ######################################################################
        



        ######################################################################
        ##############  INPUT DATA  - model 1
        ######################################################################
        #This is the time buffer (how many time samples will be used):
        buffer_length=input_chords
        #This is the ammount of the characteristics choosen starting from the all_characteristics index:
        characteristics_buff=89
        #This is the len of the characteristics: 
        #(choose 89 if duration should be also considered)
        #(choose 88 if duration shouldn't be considered)
        #(choose 41 to include characteristics between 41-characteristics_buff:41
        all_characteristics=89
        inputs=np.full((len(all_notes_new)-buffer_length-output_predicted_chords,buffer_length,characteristics_buff), 0.0)
        
        # Separate to inputs-outputs
        for time_index in range(len(all_notes_new)-buffer_length-output_predicted_chords): #number of changes
            input_temp=all_notes_new[time_index:time_index+buffer_length,all_characteristics-characteristics_buff:all_characteristics]
            inputs[time_index,0:buffer_length,0:characteristics_buff]=(input_temp)
        
        # len(all_notes) samples, each with buffer_length time steps and 89 features
        input_data1_new = inputs.copy()
        
        
        ######################################################################
        ##############  OUTPUT DATA
        ######################################################################

        output_data1_new=[]
        num_samples = len(all_notes_new) - buffer_length-output_predicted_chords  # Adjust for the two future values
        for i in range(89):
            #This is the time buffer (how many time samples will be used):
            buffer_length=input_chords
            outputs=np.full((len(all_notes_new)-buffer_length,1,1), 0.0)
            
            SPESIFIC_CHARACTERISTIC_TO_PREDICT = i
            
            outputs = np.full((num_samples, output_predicted_chords, 1), 0.0)
            for time_index in range(num_samples):
                output_temp = all_notes_new[time_index + buffer_length:time_index + buffer_length + output_predicted_chords, i]
                for k in range(output_predicted_chords):
                    outputs[time_index, k, 0] = output_temp[k]
            output_data1_new.append(outputs)
        ################################################################################
        ################################################################################   

        

        ################################################   example input 3 (training song's pattern)
        #initialize:
        user_input_time_step = []
        for i in range(buffer_length1):
            user_input_time_step.append(np.zeros(89))
        
        # add some random notes (with hit power). 
        index_check = 26
        user_input_time_step =  input_data1_new[index_check,:,:]
    
    

        
        normalized_user_input_time_step = np.array(user_input_time_step.copy())

        '''
        ALREADY NORMALIZED...THUS GO TO PREDICT DIRECTLY
        PREDICT............................
        ##################################
        '''
                    
        
        ##### Predict:   # AI model 1
        normalized_user_input_time_step_reshaped = normalized_user_input_time_step.reshape((1, buffer_length1, characteristics_buff))    
        predictNextSequence=[]
        for i in range(0,89): #for each characteristic
            temp = lmodels[i].predict(normalized_user_input_time_step_reshaped)
            temp[temp<0.01] = 0
            predictNextSequence.append(temp)
            

        if True:
            # also chsck initial output data:
            index_check = index_check
            real_notes_new = []    
            for i in range(89):    
                real_notes_new.append(output_data1_new[i][index_check])
                



        ##### De-normalize Prediction:
        ####################### Data De-Normalization ################################
        # explicit function to denormalize array
        def denormalize(arr, t_min, t_max):
            diff = 1
            diff_arr = t_max - t_min  
            if np.isnan(diff_arr) == False:
                    temp = t_min+((arr-0)*diff_arr/diff)
            return temp
            


        final_predicted_numpy_actual = copy.deepcopy(predictNextSequence)
        #De-Normalize each feature indipendently:
        for i in range(89):
            for k in range(output_predicted_chords):            
                value = (final_predicted_numpy_actual[i][0][k])
                if True:
                        final_predicted_numpy_actual[i][0][k] = denormalize(value,
                                                    characteristics_min_new[i],
                                                    characteristics_max_new[i])
                   
 

        final_real_notes_numpy_actual = copy.deepcopy(real_notes_new)
        #De-Normalize each feature indipendently:
        for i in range(89):
            for k in range(output_predicted_chords):
                value = (final_real_notes_numpy_actual[i][k])
                if True:
                        final_real_notes_numpy_actual[i][k] = denormalize(value,
                                                    characteristics_min_new[i],
                                                    characteristics_max_new[i])


        flattened = [np.squeeze(arr) for arr in final_predicted_numpy_actual]
        stacked = np.array(flattened)
        flattened_final_predicted_numpy_actual = (stacked.T).astype(np.int32)

        flattened = [np.squeeze(arr) for arr in final_real_notes_numpy_actual]
        stacked = np.array(flattened)
        flattened_final_real_numpy_actual = (stacked.T).astype(np.int32)
        
        
        
        
        final_predicted_numpy_actual_=flattened_final_predicted_numpy_actual.copy()
        final_predicted_numpy_actual_=np.array(final_predicted_numpy_actual_)    
        final_predicted_numpy_actual_[np.isnan(final_predicted_numpy_actual_)] = 0
        final_predicted_numpy_actual_[final_predicted_numpy_actual_>127] = 127
        final_predicted_numpy_actual_[final_predicted_numpy_actual_<skip_notes_lower_than] = 0
        final_predicted_numpy_actual_=np.round(final_predicted_numpy_actual_)   
        
                
        reconstructed_final_predicted_numpy_actual = reconstruct_array(final_predicted_numpy_actual_)
        reconstructed_final_real_numpy_actual = reconstruct_array(flattened_final_real_numpy_actual)
            
            
            
        mid_reconstructed_final_predicted_numpy_actual = arry2mid(reconstructed_final_predicted_numpy_actual, 545455)
        mid_reconstructed_final_predicted_numpy_actual.save(str(output_dir)+'/reconstructed_final_predicted_numpy_actual.mid')
          
          
        mid_reconstructed_final_real_numpy_actual = arry2mid(reconstructed_final_real_numpy_actual, 545455)
        mid_reconstructed_final_real_numpy_actual.save(str(output_dir)+'/reconstructed_final_real_numpy_actual.mid')
          
        #for debugging return predicted sequence of model 1
        return predictNextSequence    
        return real_notes_new

###########################################################
# 2) Put your existing "example == 2" code into a function
###########################################################
def run_example_2(skip_notes_lower_than):
       global predictNextSequence
       global real_notes

       ################################################   example input 3 (training song's pattern)
       #initialize:
       user_input_time_step = []
       for i in range(buffer_length1):
           user_input_time_step.append(np.zeros(89))
       
       # add some random notes (with hit power). 
       index_check = 26
       user_input_time_step =  input_data1[index_check,:,:]
   
   
   
       
       normalized_user_input_time_step = np.array(user_input_time_step.copy())

       '''
       ALREADY NORMALIZED...THUS GO TO PREDICT DIRECTLY
       PREDICT............................
       ##################################
       '''
                   
       
       ##### Predict:   # AI model 1
       normalized_user_input_time_step_reshaped = normalized_user_input_time_step.reshape((1, buffer_length1, characteristics_buff))    
       predictNextSequence=[]
       for i in range(0,89): #for each characteristic
           temp = lmodels[i].predict(normalized_user_input_time_step_reshaped)
           temp[temp<0.01] = 0
           predictNextSequence.append(temp)
           

       if True:
           # also chsck initial output data:
           index_check = index_check
           real_notes = []    
           for i in range(89):    
               real_notes.append(output_data1[i][index_check])
               



       ##### De-normalize Prediction:
       ####################### Data De-Normalization ################################
       # explicit function to denormalize array
       def denormalize(arr, t_min, t_max):
           diff = 1
           diff_arr = t_max - t_min  
           if np.isnan(diff_arr) == False:
                   temp = t_min+((arr-0)*diff_arr/diff)
           return temp
           


       final_predicted_numpy_actual = copy.deepcopy(predictNextSequence)
       #De-Normalize each feature indipendently:
       for i in range(89):
           for k in range(output_predicted_chords):            
               value = (final_predicted_numpy_actual[i][0][k])
               if True:
                       final_predicted_numpy_actual[i][0][k] = denormalize(value,
                                                   characteristics_min[i],
                                                   characteristics_max[i])
                  
  


       final_real_notes_numpy_actual = copy.deepcopy(real_notes)
       #De-Normalize each feature indipendently:
       for i in range(89):
           for k in range(output_predicted_chords):
               value = (final_real_notes_numpy_actual[i][k])
               if True:
                       final_real_notes_numpy_actual[i][k] = denormalize(value,
                                                   characteristics_min[i],
                                                   characteristics_max[i])


       
       flattened = [np.squeeze(arr) for arr in final_predicted_numpy_actual]
       stacked = np.array(flattened)
       flattened_final_predicted_numpy_actual = (stacked.T).astype(np.int32)
       
       
       flattened = [np.squeeze(arr) for arr in final_real_notes_numpy_actual]
       stacked = np.array(flattened)
       flattened_final_real_numpy_actual = (stacked.T).astype(np.int32)
       
       
       
       
       final_predicted_numpy_actual_=flattened_final_predicted_numpy_actual.copy()
       final_predicted_numpy_actual_=np.array(final_predicted_numpy_actual_)    
       final_predicted_numpy_actual_[np.isnan(final_predicted_numpy_actual_)] = 0
       final_predicted_numpy_actual_[final_predicted_numpy_actual_>127] = 127
       final_predicted_numpy_actual_[final_predicted_numpy_actual_<skip_notes_lower_than] = 0
       final_predicted_numpy_actual_=np.round(final_predicted_numpy_actual_)   
       


               
       reconstructed_final_predicted_numpy_actual = reconstruct_array(final_predicted_numpy_actual_)
       reconstructed_final_real_numpy_actual = reconstruct_array(flattened_final_real_numpy_actual)
           
           
           
       mid_reconstructed_final_predicted_numpy_actual = arry2mid(reconstructed_final_predicted_numpy_actual, 545455)
       mid_reconstructed_final_predicted_numpy_actual.save(str(output_dir)+'/reconstructed_final_predicted_numpy_actual.mid')
         
         
       mid_reconstructed_final_real_numpy_actual = arry2mid(reconstructed_final_real_numpy_actual, 545455)
       mid_reconstructed_final_real_numpy_actual.save(str(output_dir)+'/reconstructed_final_real_numpy_actual.mid')
         
       #for debugging return predicted sequence of model 1
       return predictNextSequence       
       return real_notes

###############################################################################
# 4) A mapping of instrument name -> GM program number
###############################################################################
INSTRUMENTS = {
    "Acoustic Grand Piano (0)": 0,
    "Church Organ (19)": 19,
    "Electric Guitar (jazz) (26)": 26,
    "Acoustic Guitar (nylon) (24)": 24,
    "Violin (40)": 40,
    "SynthBrass 1 (80)": 80
}




if mode == "option_1":
    run_example_1(str(input_path),skip_notes_lower_than)
if mode == "option_2":
    run_example_2(str(input_path),skip_notes_lower_than)




