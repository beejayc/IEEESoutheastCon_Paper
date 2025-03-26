import os
import json
import pandas as pd
import numpy as np



#------------------------------------------------------------------------------------
''' Prompt Messages used for Few Shot Spectrogram Analysis '''
#------------------------------------------------------------------------------------
def get_prompt_msgs_study(promptkey):
    
    prompt_msgs = {
        "fewshot_setobj"      : r"You are an expert scientific spectogram analyzer. Provided below are examples with tag ##EXAMPLE## are spectrogram corresponding to different defect classes with a tag ##DEFECT CLASS ##. Order given will be image first, followed by defect class text description. Given a plot of spectrogram images with x axis in time and y axis in frequeny. Amplitude of the signals are color coded in Viridis colormap scale. Extract key spectrogram information such as peak frequencies, peak amplitudes, frequency band characteristics, any harmonic patterns or any other relevant time spectrogram features. Eliminate any frequency analysis above 6000 Hz. Dont hallucinate. Extract information strictly from the plot and associate it with the defect class given after the image",
        "run_inference_sample": r"Now, given a new spectrogram , analyze it considering factors such as frequency patterns,intensity and time variations. Focus solely on the patterns presented in the spectrogram. Do not let any assumptions or environmental settings influence your decision. Your task is to determine which of  the ##EXAMPLE##  classes the new spectrogram most closely resembles . No explanations needed in the response. Your response must contain only the exact name of the class in the format PREDICTED CLASS: [Class Name]" 
    }
    
    
    return prompt_msgs[promptkey]
#------------------------------------------------------------------------------------

# "fewshot_setobj"      : r"You are an expert scientific spectogram analyzer. Provided below are examples with tag ##EXAMPLE## are spectrogram corresponding to different defect classes with a tag ##DEFECT CLASS ##. Given a plot of spectrogram images with x axis in time and y axis in frequeny. Amplitude of the signals are color coded in Viridis colormap scale. Extract key spectrogram information such as peak frequencies, peak amplitudes, frequency band characteristics, any harmonic patterns or any other relevant time spectrogram features. Eliminate any frequency analysis above 6000 Hz. Dont hallucinate. Extract information strictly from the plot and associate it with the defect class given after the image",