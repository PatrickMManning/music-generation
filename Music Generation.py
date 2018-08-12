
# coding: utf-8

# # Music Generation

# In[1]:

import pyaudio
import math
from numpy import fromstring,linspace,sin,pi,int8,int16,rint,sign,arcsin,tan,arctan,cos,append,multiply,add,subtract,divide,repeat,random,clip,fft,zeros,cumsum,minimum,concatenate,transpose,ascontiguousarray,vstack,copy,tile
import time
import pandas as pd
from enum import Enum
from scipy.io import wavfile
from scipy.signal import stft,istft
import sounddevice as sd


# In[2]:

get_ipython().magic('matplotlib inline')


# In[3]:

#p = pyaudio.PyAudio()
#for i in range(p.get_device_count()):
#    dev = p.get_device_info_by_index(i)
#    print((i,dev['name'],dev['maxInputChannels']))
#
#stream = p.open(output=True, channels=2, rate=RATE, format=pyaudio.paInt16, output_device_index=1)


# ## Constants

# In[4]:

RATE=44100
BEAT_128 = 0.46875


# In[49]:

stream = sd.OutputStream(samplerate=RATE, dtype=int16, channels=2)
stream.start()


# ## Utilities

# In[6]:

class Note(Enum):
    C3 = 131
    D3 = 147
    E3 = 165
    F3 = 175
    G3 = 196
    A4 = 220
    B4 = 247
    C4 = 262


# In[7]:

def inspect_sound(sound_data):
    df = pd.DataFrame(sound_data)
    print("shape: ", df.shape)
    print("head: ")
    df.head(101).plot()
    stream.write(df.values)


# ## Signal Generators

# In[8]:

def wave(frequency=Note.C3.value, length=1, amplitude=8000, phase=0, sample_rate=RATE, function=None, function_args=None):
    time = linspace(0,length,length*sample_rate)
    wavelength = 1/frequency
    data = function(time, wavelength, amplitude, phase, function_args)
    #data = data.astype(int16) # two byte integers
    data = mono_to_stereo(data)
    return data.astype(int16)


# In[9]:

def sine_function(time, wavelength, amplitude, phase, function_args=None):
    return amplitude * sin((2*pi*time - phase)/wavelength)


# In[10]:

def square_function(time, wavelength, amplitude, phase, function_args=None):
    return amplitude * sign(sin((2*pi*time - phase)/wavelength))


# In[11]:

def triangle_function(time, wavelength, amplitude, phase, function_args=None):
    return (2*amplitude/pi) * arcsin(sin((2*pi*time - phase)/wavelength))


# In[12]:

def sawtooth_function(time, wavelength, amplitude, phase, function_args=None):
    return (2*amplitude/pi) * arctan(tan((2*pi*time - phase)/(2*wavelength)))


# In[13]:

def inverse_sawtooth_function(time, wavelength, amplitude, phase, function_args=None):
    return (2*amplitude/pi) * arctan(tan((2*pi*time - phase)/(2*wavelength))) * -1


# In[14]:

def pulse_function(time, wavelength, amplitude, phase, function_args):
    pulse_width = function_args['pulse_width']
    return amplitude * sign(sin((2*pi*time - phase)/wavelength)-(1-pulse_width))


# In[15]:

def white_noise_function(time, wavelength, amplitude, phase, function_args):
    return random.random(len(time))*amplitude


# In[16]:

def silence(time, wavelength, amplitude, phase, function_args):
    return zeros(len(time))


# In[17]:

def identity(signal):
    return signal


# In[144]:

def frequency_modulation_synthesizer_sine(carrier=Note.C3.value, length=1, amplitude=5000, sample_rate=RATE, modulation_depth=0, modulation_frequency=0, carrier_phase=0, modulator_phase=0):
    time = linspace(0,length,length*sample_rate)
    signal = (amplitude*sin(2*pi*time*carrier + modulation_depth*sin(2*pi*time*modulation_frequency-modulator_phase) - carrier_phase))
    signal = mono_to_stereo(signal)
    return signal.astype(int16)


# In[145]:

def frequency_modulation_synthesizer(carrier=Note.C3.value, carrier_function=sine_function, length=1, amplitude=5000, sample_rate=RATE, modulation_depth=0, modulation_frequency=0, modulation_function=sine_function, carrier_phase=0, modulator_phase=0):
    time = linspace(0,length,length*sample_rate)
    if modulation_frequency == 0:
        modulator = time
    else:
        modulator = modulation_function(time, 1/modulation_frequency, modulation_depth, modulator_phase)
    signal = carrier_function(modulator, 1/carrier, amplitude, carrier_phase)
    signal = mono_to_stereo(signal)
    return signal.astype(int16)


# In[19]:

def sampler(path):
    sample = wavfile.read(path)[1]
    if sample.ndim == 1:
        return mono_to_stereo(sample).astype(int16)
    elif sample.ndim == 2:
        return sample.astype(int16)
    else:
        raise ValueError('Sound has too many dimensions.  Only mono (_, 1) and stereo (_, 2) are supported.')


# ## Plugins

# In[20]:

def linear_asdr_envelope(sound, attack, decay, sustain, release):
    if sound.ndim == 1:
        return linear_asdr_envelope_mono(sound, attack, decay, sustain, release)
    elif sound.ndim == 2:
        return linear_asdr_envelope_stereo(sound, attack, decay, sustain, release)
    else:
        raise ValueError('Sound has too many dimensions.  Only mono (_, 1) and stereo (_, 2) are supported.')

def linear_asdr_envelope_mono(sound, attack, decay, sustain, release):
    peak = sound.max()
    a = linspace(0,1,RATE*attack)
    d = linspace(1,sustain,RATE*decay)
    s = linspace(sustain,sustain,len(sound)-(RATE*(attack+decay+release)))
    r = linspace(sustain,0,RATE*release)
    envelope = append(append(a,d),append(s,r))
    return multiply(envelope,sound).astype(int16)

def linear_asdr_envelope_stereo(sound, attack, decay, sustain, release):
    peak = sound.max()
    a = linspace(0,1,RATE*attack)
    d = linspace(1,sustain,RATE*decay)
    s = linspace(sustain,sustain,len(sound)-(RATE*(attack+decay+release)))        
    r = linspace(sustain,0,RATE*release)
    envelope = append(append(a,d),append(s,r))
    envelope = mono_to_stereo(envelope)
    return multiply(envelope,sound).astype(int16)


# In[21]:

def time_warp(sound, coefficient):
    if sound.ndim == 1:
        return time_warp_mono(sound, coefficient)
    elif sound.ndim == 2:
        return time_warp_stereo(sound, coefficient)
    else:
        raise ValueError('Sound has too many dimensions.  Only mono (_, 1) and stereo (_, 2) are supported.')

def time_warp_mono(sound, coefficient):
    if coefficient >= 1:
        return repeat(sound, coefficient)
    else:
        slicer = int(1 / coefficient)
        return sound[0::slicer].astype(int16)

def time_warp_stereo(sound, coefficient):
    if coefficient >= 1:
        return repeat(sound, coefficient, axis=0)
    else:
        slicer = int(1 / coefficient)
        return sound[0::slicer].astype(int16)


# In[22]:

def lfo(sound, frequency, amount=1, wave_function=sine_function, wave_function_args={}):
    lfo = wave(function=wave_function, frequency=frequency, length=int(len(sound)/RATE), function_args=wave_function_args)
    lfo = add(lfo, lfo.max()) # shift lfo to positive
    lfo = divide(lfo,lfo.max()) # Make lfo range 0 to 1
    lfo = multiply(lfo,amount) # scale lfo by amount
    lfo = add(lfo, (1-amount)) # shift lfo up so peak is at 1 and min is (1-amount)
    return multiply(lfo,sound).astype(int16)


# In[23]:

def arpeggiator(sound, step_size, frequency):
    return lfo(sound, frequency=frequency, wave_function=pulse_function, wave_function_args={'pulse_width':step_size}).astype(int16)


# In[24]:

def distortion(sound, amount):
    sound = multiply(sound, amount)
    amount = 1 / amount
    return clip(sound, int(sound.min()*amount), int(sound.max()*amount)).astype(int16)


# In[25]:

def utility(sound, amount):
    return multiply(sound, amount).astype(int16)


# In[26]:

def limiter(signal, amount):
    signs = sign(signal)
    return multiply(minimum(abs(signal),signal.max() * amount), signs).astype(int16)


# In[27]:

def offset(sound, length=0):
    shift_amount = int(min(abs(length*RATE), len(sound)))
    shifted = zeros(sound.shape)
    if length >=0:
        shifted[shift_amount:] = sound[0:len(sound)-shift_amount]
    else:
        shifted[0:shift_amount] = sound[len(sound) - shift_amount:]
    return shifted.astype(int16)


# In[28]:

def moving_average_low_pass_filter(sound, periods):
    max_amplitude = sound.max()
    cumulative = cumsum(sound)
    cumulative[periods:] = cumulative[periods:] - cumulative[:-periods]
    unadjusted = cumulative[periods - 1:] / periods
    louder = (max_amplitude / unadjusted.max()) * unadjusted
    return (louder).astype(int16)


# In[29]:

def stft_low_pass_filter(sound, cutoff, amount=0):
    f, t, Zxx = stft(sound)
    for i,x in enumerate(Zxx.real):
        if i > cutoff:
            Zxx[i] = multiply(Zxx[i],zeros(x.shape)+amount)
    return istft(Zxx)[1].astype(int16)


# In[30]:

def stft_high_pass_filter(sound, cutoff, amount=0):
    f, t, Zxx = stft(sound)
    for i,x in enumerate(Zxx.real):
        if i < cutoff:
            Zxx[i] = multiply(Zxx[i],zeros(x.shape)+amount)
    return istft(Zxx)[1].astype(int16)


# In[31]:

def stft_band_pass_filter(sound, cutoff_lo, cutoff_hi, amount=0):
    f, t, Zxx = stft(sound)
    for i,x in enumerate(Zxx.real):
        if i < cutoff_lo or i > cutoff_hi:
            Zxx[i] = multiply(Zxx[i],zeros(x.shape)+amount)
    return istft(Zxx)[1].astype(int16)


# In[32]:

def add_waves(sound1, sound2):
    if len(sound1) < len(sound2):
        combined = sound2.copy()
        combined[:len(sound1)] += sound1
    else:
        combined = sound1.copy()
        combined[:len(sound2)] += sound2
    return combined.astype(int16)


# In[33]:

def fade_out(sound, amount=None, fade_function=linspace):
    if sound.ndim == 1:
        return fade_out_mono(sound, amount=amount, fade_function=fade_function)
    elif sound.ndim == 2:
        return fade_out_stereo(sound, amount=amount, fade_function=fade_function)
    else:
        raise ValueError('Sound has too many dimensions.  Only mono (_, 1) and stereo (_, 2) are supported.')

def fade_out_mono(sound, amount=None, fade_function=linspace):
    if amount is None:
        fade_length = 10
    else:
        fade_length = len(sound) * amount
    fade_length = min(fade_length, len(sound))
    fade_length = int(fade_length)
    fade_amount = linspace(1,0,fade_length)
    padding = add(zeros(len(sound) - fade_length),1)
    fade_amount = concatenate((padding, fade_amount))
    return multiply(sound, fade_amount).astype(int16)

def fade_out_stereo(sound, amount=None, fade_function=linspace):
    if amount is None:
        fade_length = 10
    else:
        fade_length = len(sound) * amount
    fade_length = min(fade_length, len(sound))
    fade_length = int(fade_length)
    fade_amount = linspace(1,0,fade_length)
    fade_amount = mono_to_stereo(fade_amount)
    padding = add(zeros(len(sound) - fade_length),1)
    padding = mono_to_stereo(padding)
    fade_amount = concatenate((padding, fade_amount))
    return multiply(sound, fade_amount).astype(int16)


# In[34]:

def fade_in(sound, amount=None, fade_function=linspace):
    if sound.ndim == 1:
        return fade_in_mono(sound, amount=amount, fade_function=fade_function)
    elif sound.ndim == 2:
        return fade_in_stereo(sound, amount=amount, fade_function=fade_function)
    else:
        raise ValueError('Sound has too many dimensions.  Only mono (_, 1) and stereo (_, 2) are supported.')

def fade_in_mono(sound, amount=None, fade_function=linspace):
    if amount is None:
        fade_length = 10
    else:
        fade_length = len(sound) * amount
    fade_length = min(fade_length, len(sound))
    fade_length = int(fade_length)
    fade_amount = linspace(0,1,fade_length)
    padding = add(zeros(len(sound) - fade_length),1)
    fade_amount = concatenate((fade_amount,padding))
    return multiply(sound, fade_amount).astype(int16)

def fade_in_stereo(sound, amount=None, fade_function=linspace):
    if amount is None:
        fade_length = 10
    else:
        fade_length = len(sound) * amount
    fade_length = min(fade_length, len(sound))
    fade_length = int(fade_length)
    fade_amount = linspace(0,1,fade_length)
    fade_amount = mono_to_stereo(fade_amount)
    padding = add(zeros(len(sound) - fade_length),1)
    padding = mono_to_stereo(padding)
    fade_amount = concatenate((fade_amount,padding))
    return multiply(sound, fade_amount).astype(int16)


# In[35]:

def fade(sound, amount=None, fade_function=linspace):
    return (fade_out(fade_in(sound, amount=amount, fade_function=fade_function),amount=amount, fade_function=fade_function)).astype(int16)


# In[36]:

def mono_to_stereo(sound):
    return ascontiguousarray(transpose(vstack((sound,sound))))


# In[37]:

def pad(sound, length):
    length *= RATE
    length = int(length)
    if len(sound) == length:
        return sound
    elif len(sound) > length:
        return sound[:length]
    else:
        remaining = length - len(sound)
        return concatenate((sound, mono_to_stereo(zeros(remaining)))).astype(int16)


# In[38]:

def balance(sound, l_r_balance=0.5):
    if l_r_balance > 1:
        l_r_balance = 1
    elif l_r_balance < 0:
        l_r_balance = 0
    
    if sound.ndim == 1:
        return balance_mono(sound, l_r_balance)
    elif sound.ndim == 2:
        return balance_stereo(sound, l_r_balance)
    else:
        raise ValueError('Sound has too many dimensions.  Only mono (_, 1) and stereo (_, 2) are supported.')

def balance_mono(sound, l_r_balance):
    return sound

def balance_stereo(sound, l_r_balance):
    sound = copy(sound) # making function pure by not mutating original sound
    l_r_balance *= 2
    l_balance = 2 - l_r_balance
    r_balance = l_r_balance
    sound[:,0] = multiply(sound[:,0],l_balance)
    sound[:,1] = multiply(sound[:,1],r_balance)
    return sound.astype(int16)


# In[39]:

def phase_invert(sound):
    return sound * -1


# In[40]:

def loop(sound, loop_count):
    if sound.ndim == 1:
        return loop_mono(sound, loop_count)
    elif sound.ndim == 2:
        return loop_stereo(sound, loop_count)
    else:
        raise ValueError('Sound has too many dimensions.  Only mono (_, 1) and stereo (_, 2) are supported.')

def loop_mono(sound, loop_count):
    return tile(sound, loop_count).astype(int16)

def loop_stereo(sound, loop_count):
    return tile(sound, (loop_count,1)).astype(int16)


# In[812]:

def echo(sound, delay=0, decay_length=1, decay_amount=0.01, decay_function=geomspace):
    delay = (len(sound) + ((RATE*delay) - len(sound))) / RATE
    pluck = pad(sound, delay)
    pluck = loop(pluck,decay_length)
    decay = decay_function(1,decay_amount,len(pluck))
    pluck = multiply(pluck,transpose((decay,decay)))
    return pluck.astype(int16)


# ## Instruments

# In[41]:

kick_thump = wave(function=sine_function, length=0.5, frequency=Note.C3.value/2)
kick_attack = linear_asdr_envelope(wave(function=white_noise_function, length=0.1, frequency=Note.C3.value*5),0,0.03,0,0)
#kick = add_waves(kick_thump,kick_thump)
kick = kick_thump
stream.write(kick)


# In[42]:

kick_hi_thump = linear_asdr_envelope(wave(function=sine_function, length=0.75, frequency=Note.C3.value/5),0.1,0.5,0.2,0)
kick_hi_attack = linear_asdr_envelope(wave(function=sine_function, length=0.5, frequency=Note.C3.value/2),0,0.2,0,0)
kick_hi = add_waves(kick_hi_thump,kick_hi_attack)
kick_hi = time_warp(kick_hi,0.4)
stream.write(kick_hi)


# In[43]:

rest = wave(function=silence, length=1)


# In[44]:

snare_attack = wave(function=sawtooth_function, length=0.01, frequency=Note.C3.value)
snare_sizzle = linear_asdr_envelope(wave(function=white_noise_function, length=1, frequency=Note.C3.value),0.2,0.5,0,0)
snare_sizzle = utility(snare_sizzle, 0.5)
snare_sizzle_2 = linear_asdr_envelope(wave(function=sine_function, length=1, frequency=Note.C3.value/4),0.01,0.5,0,0)
snare = add_waves(snare_attack,snare_sizzle)
snare = add_waves(snare, snare_sizzle_2)
snare = time_warp(snare,0.15)
stream.write(snare)


# In[45]:

hi_hat = linear_asdr_envelope(wave(function=white_noise_function, length=1, frequency=Note.C3.value),0,0.1,0.1,0)
hi_hat = time_warp(hi_hat, 0.2)
stream.write(hi_hat)


# ## Workspace

# In[57]:

xx = loop(pad(fade(kick_hi,amount=None),BEAT_128),4)

stream.write(xx)


# In[56]:

import mido


# In[ ]:

mid = mido.MidiFile('song.mid')
for msg in mid.play():
    port.send(msg)


# In[720]:

pluck = wave(function=sawtooth_function, length=0.0075)
p2 = echo(pluck, delay=0.009, decay_length=15, decay_amount=0.01)
p3 = echo(p2, delay=0.009, decay_length=55)
p4 = echo(p3, delay=0.05, decay_length=10)

stream.write(p4)


# In[805]:

stream.write(echo(snare, delay=0.2, decay_length=7, decay_amount=0.01))


# #### Sound replication

# In[163]:

from numpy import setdiff1d,array
from numpy.linalg import norm
from scipy.optimize import minimize, differential_evolution
from scipy.fftpack import fft
from python_speech_features import mfcc
from numpy import geomspace


# In[157]:

SOUND = sampler("D:\Pat\projects\programming\music generation\samples\Galactica - C3 - 2.wav")


# In[158]:

def function_selector(value):
    value = int(value)
    if value == 1:
        return sine_function
    elif value == 2:
        return sawtooth_function
    elif value == 3:
        return square_function
    elif value == 4:
        return triangle_function
    else:
        return silence


# In[222]:

def play_sound(c1, c2, d1, f1, d2, f2, cf1, cf2, mf1, mf2, fm2pct):
    fm1 = frequency_modulation_synthesizer(carrier=c1, length=1, modulation_depth=d1, modulation_frequency=f1, carrier_function=function_selector(cf1), modulation_function=function_selector(mf1))
    fm2 = frequency_modulation_synthesizer(carrier=c2, length=1, modulation_depth=d2, modulation_frequency=f2, carrier_function=function_selector(cf2), modulation_function=function_selector(mf2))
    fm2 = (fm2pct * fm2).astype(int16)
    fm3 = add_waves(fm1, fm2)
    return fm3


# In[292]:

def sound_replication(x0):
    c1 = x0[0]
    c2 = x0[1]
    d1 = x0[2]
    f1 = x0[3]
    d2 = x0[4]
    f2 = x0[5]
    cf1 = x0[6]
    cf2 = x0[7]
    mf1 = x0[8]
    mf2 = x0[9]
    fm2pct = x0[10]
    
    fm3 = play_sound(c1, c2, d1, f1, d2, f2, cf1, cf2, mf1, mf2, fm2pct)
    
    fft_size = 500
    s1 = fft(SOUND, fft_size, axis=0)[:,0]
    s2 = fft(fm3, fft_size, axis=0)[:,0]
    
    #s1 = mfcc(SOUND, samplerate=RATE).mean(axis=0)
    #s2 = mfcc(fm3, samplerate=RATE).mean(axis=0)
    
    ### importance_multiplier = linspace(1, 0, fft_size) ** 10
    #importance_multiplier = geomspace(1,0.01,500)
    #s1 = multiply(s1, importance_multiplier)
    #s2 = multiply(s2, importance_multiplier)
    
    # Can add a penalty on complexity.  Sum up the x0 values (carrier freqs, modulations) 
    # and add it to the result.  This will favor lower freq results.
    
    return norm(subtract(s1,s2))
    #return d1 - 0


# In[293]:

# carrier_freq1, carrier_freq2, mod_depth1, mod_freq1, mod_depth2, mod_freq2, carrier_fun1, carrier_fun2, mod_fun1, mod_fun2, fm2_pct
bounds = [(30,10000),(30,10000),(1,500),(0,10),(1,500),(0,10),(1,4),(1,4),(1,4),(1,4),(0,1)]
#bounds = [(30,1000),(30,500),(1,10),(0,10),(1,10),(0,10),(1,4),(1,4),(1,4),(1,4),(0,1)]


# In[287]:

#res = minimize(sound_replication, x0, bounds=bounds, method='SLSQP', options={'disp': True})
res = differential_evolution(sound_replication, bounds=bounds)
res.x


# In[721]:

stream.write(play_sound(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], res.x[6], res.x[7], res.x[8], res.x[9], res.x[10]))


# In[281]:

stream.write(play_sound(700, 500, 8, 0.51, 0.5, 0.05, 1, 1, 1, 1, 0.8))


# In[227]:

stream.write(SOUND)


# In[234]:

pd.DataFrame(fft(SOUND, 500, axis=0)[:,0]).plot()


# In[290]:

pd.DataFrame(fft(play_sound(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], res.x[6], res.x[7], res.x[8], res.x[9], res.x[10]), 500, axis=0)[:,0]).plot()


# #### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ## Notes

# ### TODO
# - Convert every function to handle both stereo and mono (originally designed for mono)
# 
# - LFO (works on sounds but not samples?)
# - ADSR envelope
# - Filters (FFT?)
# - Arp (very simplistic right now.  Add a function to it for stepping up each note.  Pass Identity fun by default)
# - Sampler
# - How to change BPM?
#     - should it be during playback like changing the stream rate
#     - can I double every value in the array to make twice as long?
# - distortion
# - limiter
# - "utility"
# - fade in, fade out
# - mono to stereo and back
# - echo
# 
# 
# - MIDI
# - return track
# - effects rack
# - stereo spread
# - reverb
# - delay
# - crossfade
# - Oscilator (wav combiners) or combining of synths.
# - style transfer of samples (vocoder?)
# - take the tone of written word (NLP) and turn into music?

# ### HOW TO USE
# stream.write(wave(function=sine_function))
# 
# stream.write(wave(function=square_function))
# 
# stream.write(wave(function=triangle_function))
# 
# stream.write(wave(function=sawtooth_function))
# 
# stream.write(wave(function=pulse_function, function_args={'pulse_width':0.25}))
# 
# stream.write(wave(function=white_noise_function))
# 
# stream.write(linear_asdr_envelope(wave(function=square_function, length=6), 1, 1, 0.5, 1))
# 
# stream.write(wave(function=sine_function, length=3, frequency=Note.C3.value, amplitude=5000)+wave(function=sine_function, length=3, frequency=Note.C3.value*2, amplitude=2000))
# 
# stream.write(lfo(wave(function=sawtooth_function,length=9),frequency=3, amount=1))
# 
# stream.write(distortion(wave(function=sawtooth_function), 10))
# 
# stream.write(utility(wave(function=sawtooth_function, length=3), 2))
# 
# stream.write(frequency_modulation_synthesizer(length=4, modulation_depth=20, modulation_frequency=Note.C3.value))
# 
# stream.write(offest(wave(function=sine_function),0.5))
# 
# stft_low_pass_filter(signal,3,0.25)
# 
# echo(wave(function=sawtooth_function, length=0.05), delay=0.075, decay_length=15, decay_amount=0.01)
# 
# limiter(sig,0.5)
# 
# ##### Sampler
# path = "D:\\Pat\\projects\\programming\\Accent Removal\\Samples\\wavs\\samplewav.wav"
# 
# sampler("D:\Pat\projects\programming\Accent Removal\Samples\wavs\samplewav.wav")

# sig = frequency_modulation_synthesizer(length=1, modulation_depth=15, modulation_frequency=Note.C3.value+15)
# stream.write(limiter(sig,0.5))

# saw_c = wave(function=sawtooth_function, length=1, frequency=Note.C3.value)
# saw_e = wave(function=sawtooth_function, length=1, frequency=Note.E3.value)
# saw_g = wave(function=sawtooth_function, length=1, frequency=Note.G3.value)
# saw_b = wave(function=sawtooth_function, length=1, frequency=Note.B4.value)
# rest = wave(function=silence, length=1)
# 
# stream.write(saw_c)
# stream.write(saw_e)
# stream.write(saw_g)
# stream.write(saw_c)
# stream.write(saw_e)
# stream.write(saw_g)
# stream.write(saw_c)
# stream.write(saw_e)
# stream.write(saw_g)
# stream.write(saw_c)
# stream.write(saw_e)
# stream.write(saw_g)
# stream.write(rest)
# stream.write(saw_e)
# stream.write(saw_g)
# stream.write(saw_b)
# stream.write(saw_e)
# stream.write(saw_g)
# stream.write(saw_b)
# stream.write(saw_e)
# stream.write(saw_g)
# stream.write(saw_b)
# stream.write(saw_e)
# stream.write(saw_g)
# stream.write(saw_b)
