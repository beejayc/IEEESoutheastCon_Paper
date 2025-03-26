import httpx
import os
import base64

import google.generativeai as genai
from google.generativeai import caching

import datetime
import time


from PIL import Image
from io import BytesIO
import mimetypes

import pandas as pd
import mutil_prompt_engr as mprompt_engr

import json
from google.protobuf.json_format import MessageToDict  

#----------------------------------------------------------------------------------
#  Set Google Gemini Model Client
#----------------------------------------------------------------------------------
def set_model_client(MODEL_NAME):
    model = genai.GenerativeModel(model_name = MODEL_NAME)
    genai.configure(api_key="")    # set APi Key here
    return model


#----------------------------------------------------------------------------------
#  Read Image and Downsize - set image size to  400x400
#----------------------------------------------------------------------------------
def read_resize_img(img_path):
    im = Image.open(img_path)
    im1 = im.resize((400,400) , Image.LANCZOS)
    return im1


#----------------------------------------------------------------------------------
#  Code to read and resize image
#----------------------------------------------------------------------------------
def create_image_message(image_path):
    IMG_SIZE = 400

    # Open the image file in binary mode
    with open(image_path, "rb") as image_file:
        # Read the image data into a BytesIO object
        image_bytes = BytesIO(image_file.read())

    # Open the image bytes using PIL
    im = Image.open(image_bytes)

    # Resize the image
    im1 = im.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # Save the resized image back to the BytesIO object, overwriting the original data
    im1.save(image_bytes, format="JPEG")

    # Seek to the beginning of the BytesIO object to read the image data as a bytes object
    image_bytes.seek(0)

    # Encode the binary data using Base64 encoding
    base64_encoded_data = base64.b64encode(image_bytes.read())

    # Decode base64_encoded_data from bytes to a string
    base64_string = base64_encoded_data.decode('utf-8')

    # Get the MIME type of the image based on its file extension
    mime_type, _ = mimetypes.guess_type(image_path)

    # Create the image block
    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": base64_string
        }
    }

    return image_block


#------------------------------------------------------------------------------------
''' Few shot prompt message compile '''
#------------------------------------------------------------------------------------
def few_shot_msg_compile(ref_filepath, testfilepath, DEFECT_CLASSES, NUM_SHOTS, NUM_CLASSES):
    
    prompt_msg_compile  = []
    
    # Optional System Prompt - Keep or Ignore No impact on results
    System_Prompt       = r"You are an expert in analyzing time frequency spectrograms. Analyze the defect class based on visual spectrogram representation"
    prompt_msg_compile.append(System_Prompt)
    
    # Start with the prompt message top level
    prompt_msg = str(mprompt_engr.get_prompt_msgs_study("fewshot_setobj"))
        
    # User Messages Prepare
    prompt_msg_compile.append(prompt_msg)
        
    # Read the reference files and append
    imgindx = 0
    for def_class in DEFECT_CLASSES:
        # Read the reference image file
        # image_msgs.append({"type": "text", "text": prompt_msg})
                
        for shot in range(0, NUM_SHOTS):
            prompt_msg_compile.append(read_resize_img(ref_filepath[imgindx]))
            imgindx += 1
            # prompt_msg_compile.append(def_class_msg)
        def_class_msg = "Example Spectrogram" "##EXAMPLE## "  + " ##DEFECT CLASS: ##" + def_class
        prompt_msg_compile.append(def_class_msg)
        
    # Read the test file and append
    prompt_msg = str(mprompt_engr.get_prompt_msgs_study("run_inference_sample"))
    prompt_msg_compile.append(read_resize_img(testfilepath))
    prompt_msg_compile.append(prompt_msg)
    
    
    return prompt_msg_compile


#------------------------------------------------------------------------------------
''' Few shot prompt message compile '''
#------------------------------------------------------------------------------------
# def few_shot_msg_compile_cache(ref_filepath, testfilepath, DEFECT_CLASSES, NUM_SHOTS, NUM_CLASSES):
    
#     prompt_msg_compile  = []
      
#     # Start with the prompt message top level
#     prompt_msg = str(mprompt_engr.get_prompt_msgs_study("fewshot_setobj"))
        
#     # User Messages Prepare
#     prompt_msg_compile.append(prompt_msg)
        
#     # Read the reference files and append
#     imgindx = 0
#     for def_class in DEFECT_CLASSES:
#         # Read the reference image file
#         # image_msgs.append({"type": "text", "text": prompt_msg})
#         def_class_msg = "Example Spectrogram" "##EXAMPLE## "  + " ##DEFECT CLASS: ##" + def_class
        
#         for shot in range(0, NUM_SHOTS):
#             prompt_msg_compile.append(def_class_msg)
#             prompt_msg_compile.append(read_resize_img(ref_filepath[imgindx]))
#             imgindx += 1

#     return prompt_msg_compile




#------------------------------------------------------------------------------------
''' Few shot prompt message compile '''
#------------------------------------------------------------------------------------
# def few_shot_msg_inference_cache(testfilepath):
#     prompt_msg_compile = []
    
#     # Read the test file and append
#     prompt_msg = str(mprompt_engr.get_prompt_msgs_study("run_inference_sample"))
#     prompt_msg_compile.append(prompt_msg)
#     prompt_msg_compile.append(read_resize_img(testfilepath))
    
#     return prompt_msg_compile
        
if __name__ == "__main__":
        
    # Reference Files
    ref_filepath = [
        r"/home/balajic/Projects/Schaeffwork/PAPERS/IEEECon2025/dataset/poc_studydata_v2/train/ACCEPT/20220606_094536_277.F-390931-!-220606144537628.St1_Bel1.csv-1.jpg",
        r"/home/balajic/Projects/Schaeffwork/PAPERS/IEEECon2025/dataset/poc_studydata_v2/train/DENTS/20230301_160106_143.St1_Bel1.csv.jpg",
        r"/home/balajic/Projects/Schaeffwork/PAPERS/IEEECon2025/dataset/poc_studydata_v2/train/NR/20220406_155224_741.F-390931-!-220406155052500.St1_Bel1.csv-1.jpg",
        r"/home/balajic/Projects/Schaeffwork/PAPERS/IEEECon2025/dataset/poc_studydata_v2/train/RD/20220429_144654_363.F-390931-!-220429144549735.St1_Bel1.csv.jpg",
    ]
     
    testfilepath = r"/home/balajic/Projects/Schaeffwork/PAPERS/IEEECon2025/dataset/poc_studydata_v2/test/DENTS/20220406_162234_069.F-390931-!-220406162023923.St1_Bel1.csv.jpg"

    DEFECT_CLASSES  = ['ACCEPT', 'OUTER DENT', 'NEEDLE REJECT', 'RACEWAY DENT']
    NUM_SHOTS       = 1
    NUM_CLASSES     = 4
    message_input   = few_shot_msg_compile(ref_filepath, testfilepath, DEFECT_CLASSES, NUM_SHOTS, NUM_CLASSES)
    
    with open('inpmsg.txt', 'w') as f:
        f.write(str(message_input))
    
    MODEL_NAME = "gemini-1.5-flash"
    model  =  set_model_client(MODEL_NAME)
    response = model.generate_content(message_input)
    print(response)
    
    resp_dict = response.to_dict()
    
    # with open('geminires.txt', 'w') as f:
    #     json.dump(resp_dict, f, indent=4) 
    # print(response['candidates'][0]['content']['parts'][0])
    
    print(resp_dict['candidates'][0]['content']['parts'][0])
    

    
    
    
    # Test With Cache Enabled
    # genai.configure(api_key="AIzaSyCigWWmZedJ631tXw5FImoL0CF5YNfUHK8")
    # MODEL_NAME = "gemini-1.5-flash"
    # model  =  set_model_client(MODEL_NAME)
    # # message_input_cache = few_shot_msg_compile_cache(ref_filepath, testfilepath, DEFECT_CLASSES, NUM_SHOTS, NUM_CLASSES)
    
    # cache = caching.CachedContent.create(
    # model='models/gemini-1.5-flash-001',
    # display_name='Spectrogram Reference Defect Catalog', # used to identify the cache
    
    # system_instruction=(
    #         "You are an expert in analyzing time frequency spectrograms. Analyze the defect class based on visual spectrogram representation"
    # ),
    # contents=[message_input_cache],
    # ttl=datetime.timedelta(minutes=5),
    
    # )   
    
    # model = genai.GenerativeModel.from_cached_content(cached_content=cache)
    # response = model.generate_content(few_shot_msg_inference_cache(testfilepath))
    # print(message_input)
    
    # prompt2 = "Explain the spectrogram characteristics"
    # with open(testfilepath, "rb") as image_file:  # Read image as bytes
    #     image_bytes = image_file.read()

    # response3 = cached_gemini_call(model,prompt2, image_bytes)  # API call with image
    # print(response3)

    # response4 = cached_gemini_call(model, prompt2, image_bytes)  # Cache hit with image
    # print(response4)  # From cache
    
    
    
    
    
    
    
    
    

    ## write as file out
    # with open("fewshot_msgs.txt", "w") as f:
    #     f.write(str(message_input))
    
    # client = set_model_client()
    # MODEL_NAME="claude-3-5-sonnet-20241022"

    # response = client.messages.create(
    #     model=MODEL_NAME,
    #     max_tokens=200,
    #     messages=message_input,
    # )

    # print(response)
    
