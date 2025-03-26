from anthropic import Anthropic
from PIL import Image

import base64
import mimetypes

import pandas as pd
import mutil_prompt_engr as mprompt_engr
import mutil_evalmetric as evalmetrics

from PIL import Image
from io import BytesIO
import base64
import mimetypes



#----------------------------------------------------------------------------------
#  Set Anthropic Model Client
#----------------------------------------------------------------------------------
def set_model_client():
    client = Anthropic(api_key="")   # API Key for Anthropic here
    return client





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
    
    prompt_msg_compile = []
    
    # System Prompt
    # prompt_msg_compile.append( {"role": "system", 
    #                       "content": {"type": "text", "text": "You are an expert in analyzing time frequency spectrograms. Analyze the defect class based on visual spectrogram representation"}
    #                      })
    user_msg = []
    
    # Start with the prompt message top level
    prompt_msg = str(mprompt_engr.get_prompt_msgs_study("fewshot_setobj"))
    # image_msgs.append( {"role": "user", "content": {"type": "text", "text": prompt_msg}})
    
    # User Messages Prepare
    user_msg.append({"type": "text", "text": prompt_msg})
        
    # Read the reference files and append
    imgindx = 0
    for def_class in DEFECT_CLASSES:
        # Read the reference image file
        # image_msgs.append({"type": "text", "text": prompt_msg})
        def_class_msg = "##EXAMPLE## "  + " ##DEFECT CLASS: ##" + def_class
        
        for shot in range(0, NUM_SHOTS):
            user_msg.append({"type": "text", "text": def_class_msg})
            user_msg.append(create_image_message(ref_filepath[imgindx]))
            imgindx += 1

    # Read the test file and append
    prompt_msg = str(mprompt_engr.get_prompt_msgs_study("run_inference_sample"))
    user_msg.append({"type": "text", "text": prompt_msg})
    user_msg.append(create_image_message(testfilepath))
    
    prompt_msg_compile.append({"role": "user", 
                       "content":  user_msg})

    return prompt_msg_compile







#------------------------------------------------------------------------------------
''' Few shot prompt message compile - Enable Prompt Cache '''
#------------------------------------------------------------------------------------
def cache_test_API(filenamelist):

    messages = [
        {
                "role": "system",  
                "content": [
                    {"type": "text", "text": "You are an expert in analyzing time frequency spectrograms. From Spectrogram, analyze the defect class based on visual spectrogram representation"}
                ],      
                
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Your task is to analyze spectrograms, which are visual representations of the frequency spectrum over time. Extract key information like dominant power in a frequecnyc, any harmonics present, what is the dominant frequency and all information that distinctly describe the spectrogram. Given are exmaples of different bearing classes:"},
                   
                    { "type": "text", "text": "Example Spectrogram for category - CLEAN BEARING" },
                    create_image_message(filenamelist[0]),
                            
                    { "type": "text", "text": "Example Spectrogram for category - OUTER DENT BEARING" },
                    create_image_message(filenamelist[1]),
                            
                    { "type": "text", "text": "Example Spectrogram for category - NEEDLE REJECT BEARING" },
                    create_image_message(filenamelist[2]),
                            
                    { "type": "text", "text": "Example Spectrogram for category - RACEWAY DENT BEARING" },
                    create_image_message(filenamelist[3]),
                    
                    
                    {"type": "text", "text": "Now, given a new spectrogram , analyze it considering factors such as frequency patterns,intensity and time variations. Focus solely on the patterns presented in the spectrogram. Do not let any assumptions or environmental settings influence your decision. Your task is to determine which of \
                            the example classes the new spectrogram most closely resembles . Your response must contain only the exact name of the class. Predicted Class:" },
                    create_image_message(filenamelist[4]),
                     
            ],
        }
    ]
    
    return messages                
        


def run_inference_anthropic(client, MODEL_NAME):
    
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=200,
        messages=message_input,
    )

    return response


        
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
    message_input         = few_shot_msg_compile(ref_filepath, testfilepath, DEFECT_CLASSES, NUM_SHOTS, NUM_CLASSES)
    #print(message_input)

    ## write as file out
    with open("fewshot_msgs.txt", "w") as f:
        f.write(str(message_input))
    
    # client = set_model_client()
    # MODEL_NAME="claude-3-5-sonnet-20241022"

    # response = client.messages.create(
    #     model=MODEL_NAME,
    #     max_tokens=200,
    #     messages=message_input,
    # )

    # print(response)
    
