import os
import base64
import mimetypes
import numpy as np
import re

import pandas as pd
import mutil_evalmetric as evalmetrics
import mrun_antropic_mdls as run_anthropic
import mrun_gemini_mdls as run_gemini_mdls


'''
    
    Top Level File to Launch and Run Simulation 
    Support both Anthropic and Gemini Models  - Add corresponding API keys
    
'''


#------------------------------------------------------------------------------------
''' Get all file paths in a folder - Read jpg files in a folder''' 
#------------------------------------------------------------------------------------
def get_file_paths(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
    return image_paths
#------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------
''' Run Simulationb'''

#------------------------------------------------------------------------------------
if __name__ == "__main__":
        
    # Reference Files - Dictionary of Defect Classes for 1-shot inferncing test -  Add Folder Path
    ref_filepath = [
        r"/home/balajic/Projects/Schaeffwork/PAPERS/IEEECon2025/dataset/IEEE_SoutgEastConDS/train/ACCEPT/20220606_095441_847.F-390931-!-220606145443257.St1_Bel1.csv-1.jpg",
        r"/home/balajic/Projects/Schaeffwork/PAPERS/IEEECon2025/dataset/IEEE_SoutgEastConDS/train/DENTS/20230301_142437_748.St1_Bel1.csv-1.jpg",
        r"/home/balajic/Projects/Schaeffwork/PAPERS/IEEECon2025/dataset/IEEE_SoutgEastConDS/train/NR/20220406_154939_316.F-390931-!-220406154805956.St1_Bel1.csv-1.jpg",
        r"/home/balajic/Projects/Schaeffwork/PAPERS/IEEECon2025/dataset/IEEE_SoutgEastConDS/train/RD/20220429_144811_646.F-390931-!-22042914464185.St1_Bel1.csv1.jpg",
    ]
     
    testfilepath = r"/home/balajic/Projects/Schaeffwork/PAPERS/IEEECon2025/dataset/poc_studydata_v2/test/DENTS/20220406_162234_069.F-390931-!-220406162023923.St1_Bel1.csv.jpg"

    DEFECT_CLASSES  = ['ACCEPT', 'OUTER DENT', 'NEEDLE REJECT', 'RACEWAY DENT']
    NUM_SHOTS       = 1                  # Tested Upto 3 shots - Adjust Prompte Cache Setting for Higher Order K
    NUM_CLASSES     = 4                 
    NUM_SAMPLES     = 100                # Run Random Samples of 100 images for Testing
    
    # Put all the Test Set images in the folder
    BASEDIR        = r"/home/balajic/Projects/Schaeffwork/PAPERS/IEEECon2025/dataset/IEEE_SoutgEastConDS/test"
    
    DEFECT_FOLDERS = os.listdir(BASEDIR)
    DEFECT_FOLDERS = sorted(DEFECT_FOLDERS)
        
        
    ##------------------------------
    # Run Anthropic Simulation 
    ##------------------------------
    random_seed      = 42
    confusion_matrix = np.zeros([NUM_CLASSES,NUM_CLASSES],dtype = np.int32)
    
    SIM_BASE          = "GEMINI"
    
    if SIM_BASE =="ANTHROPIC" :    
        MODEL_NAME       = "claude-3-opus-20240229"    # "claude-3-5-sonnet-20241022" 
    else:
        MODEL_NAME       = "gemini-2.0-flash"          # gemini-1.5-pro
    
    for rowid in range(0,NUM_CLASSES):
        # Word to Search
        word_to_search = DEFECT_CLASSES[rowid]

        # Regex pattern to search for the word
        pattern = rf"\b{word_to_search}\b"
        print(word_to_search)
                
        filepaths = get_file_paths(os.path.join(BASEDIR,DEFECT_FOLDERS[rowid])) 
        df_filepath = pd.DataFrame(columns=['filepaths'], data = filepaths)
        
        # Shuffle the rows
        shuffled_df = df_filepath.sample(frac=1, random_state=random_seed)
        # print(shuffled_df.iloc[0:5]['filepaths'])
        
        # Take only first 50 samples and fill in the Confusion matrix
        shuffled_df = shuffled_df[0:NUM_SAMPLES]
        
        if SIM_BASE =="ANTHROPIC":
            client      = run_anthropic.set_model_client()
        else:
            client      = run_gemini_mdls.set_model_client(MODEL_NAME)
                
        for filenum in range(0,NUM_SAMPLES):
            testfilepath   = shuffled_df.iloc[filenum]['filepaths']
            print(testfilepath)
            
            if SIM_BASE =="ANTHROPIC" :
                message_input  = run_anthropic.few_shot_msg_compile(ref_filepath, testfilepath, DEFECT_CLASSES, NUM_SHOTS, NUM_CLASSES)
                
                response = client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=200,
                    messages=message_input,
            )
                predicted_result =  response.content[0].text

            else:
                message_input  = run_gemini_mdls.few_shot_msg_compile(ref_filepath, testfilepath, DEFECT_CLASSES, NUM_SHOTS, NUM_CLASSES)
                response = client.generate_content(message_input)
                resp_dict = response.to_dict()
                predicted_result =  resp_dict['candidates'][0]['content']['parts'][0]['text']
                
            
            # Run Files
            print("True Class is:", pattern)
            print("Predicted Result is:", predicted_result)
                        
            # Search for the word in the API response message
            match = re.search(pattern, predicted_result)
            print("Match Value is:", match)

            if match is not None:
                print('Updating Matched CF')
                confusion_matrix[rowid][rowid] += 1
            else:
                for colid in range(0,NUM_CLASSES):
                    word_to_search = DEFECT_CLASSES[colid]
                    pattern = rf"\b{word_to_search}\b"
                    match = re.search(pattern, predicted_result)
                    if match is not None:
                        print("Column ID UnMatched:",colid)
                        confusion_matrix[rowid][colid] += 1
                        print('Updating UnMatched CF')
                        print(confusion_matrix)

    # Run Evaluation Metrics       
    print(confusion_matrix)
    macro_precision, macro_recall, macro_f1   =  evalmetrics.calculate_macro_scores(confusion_matrix)
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    
        
    
    # message_input         = run_anthropic.few_shot_msg_compile(ref_filepath, testfilepath, DEFECT_CLASSES, NUM_SHOTS, NUM_CLASSES)
    # run_inference_anthropic(client, MODEL_NAME):
    # #print(message_input)

    # ## write as file out
    # with open("fewshot_msgs.txt", "w") as f:
    #     f.write(str(message_input))
    
    # # client = set_model_client()
    # # MODEL_NAME="claude-3-5-sonnet-20241022"

     # response = client.messages.create(
     #     model=MODEL_NAME,
     #     max_tokens=200,
     #     messages=message_input,
     # )










