#3 Oct update
#Using openi Whispher and LLM gpt-o-mini

#Openai LLM
import openai
from openai import OpenAI

import os
import streamlit as st
import torch
import datetime
import json
import pandas as pd  

#for the dialog form for folder path
import tkinter as tk
from tkinter import filedialog



#parameter for openai


## Set the API key and model name

API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxx"

GPT_MODEL="gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", API_KEY))

dialog=""

def openai_run(model_engine, prompt_template):
    messages=[{'role':'user', 'content':f"{prompt_template}"}]

    completion = client.chat.completions.create(
    model=model_engine,
    messages=messages,
    temperature=0,)
    
    # extracting useful part of response
    response = completion.choices[0].message.content
    
    # printing response
    print(response)
    return response


def save_audio_file(audio_bytes, file_name):
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # file_name = "./"+ f"audio_{timestamp}.{file_extension}"
    # file_name =  f"audio_temp.{file_extension}"

    with open(file_name, "wb") as f:
        f.write(audio_bytes)
        f.close()
    print("*", file_name)
    return file_name

def convert_audio_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    wav_file = audio_file.name.split(".")[0] + ".wav"
    audio.export(wav_file, format="wav")
    return wav_file

def speech_to_text(audio_file):
    #print into dialog format
    dialog =""

    # Transcribe the audio
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_file, "rb"),
    )
    ## OPTIONAL: Uncomment the line below to print the transcription
    print("Transcript: ", transcription.text + "\n\n")
    
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
        {"role": "system", "content":"""You are generating a transcript with speakers label. The audio is the conversation between a telemarketer and customer. Out the transcription in a dialogue format with speaker label. If the consersation is in other languages, translate to english transcript with speakers label."""},
        {"role": "user", "content": [
            {"type": "text", "text": f"The audio transcription is: {transcription.text}"}
            ],
        }
        ],
        temperature=0,
    )
    dialog = response.choices[0].message.content
    
    return dialog



#Stage 1 
prompt1 = f"""
Objective:
Your task is to audit the conversation between a telemarketer from IPP or IPPFA and a customer. The audit will assess whether the telemarketer adhered to specific criteria during the dialogue.

Instructions:
Carefully review the provided conversation transcript and evaluate the telemarketer's compliance based on the audit criteria delimited by triple backticks. 
For each criterion, check through the conversation detail provide a detailed assessment with result, including reasons dervied only from the given conversation for the result given.
All evaluation of the compliance strictly based on the conservation content. Don't hallucination. All given Criteria must be audited.


Conversation:
{dialog}


Audit Criteria:
``` 
-1. Did the telemarketer mention their name, state they are calling from IPPFA or IPP and, state that IPPFA is a Licensed Financial Adviser providing advice on life insurance, general insurance, and CIS?
-2. Did the customer asked where they are calling from? If Yes, check telemarketer must not mentioning calling on behalf of any insurer(eg. NTUC, HSBC, Prudential, AIG etc)?
-3. Did the customer asked where did telemarketer obtain their telephone number?If Yes, Check did the telemarketer mentioned the person name who they obtain the customer's contact from?
-4. Check telemarketer MUST NOT mention that products/investment/promotion with any return value or returns are guaranteed or there is capital guarantee?
-5. Check the telemarketer is polite,conduct themselves professionally to the customer and understand customer requirements?
-6. Did the telemarketer successfully engaged the customer with a meeting/zoom session/meetup/discuss?
``` 
Return a list of JSON object that provide assessments for each of the criteria. Each  JSON object should contain the following keys:
*Question: State the criterion being evaluated. examples, "Did the telemarketer introduce their own name, mention they are calling from IPPFA or IPP, and state that IPPFA is a Licensed Financial Adviser providing advice on life insurance, general insurance, and CIS?"
*Reason: Explain by pointing out specific reason based on the information in the conversation. examples, "The telemarketer introduced themselves with name but mentioned they are calling from IPPFA"
*Result: Indicate whether the criterion was met with "Pass," "Fail," or "Not Applicable."
All given Criteria must be audited.

Output:

"""

#stage 2 
prompt2 = f"""
Objective:

Your task is to audit the conversation between a telemarketer from IPP or IPPFA and a customer. The audit will assess whether the telemarketer adhered to specific criteria during the dialogue.

Instructions:

Carefully review the provided conversation transcript and evaluate the telemarketer's compliance based on the audit criteria delimited by triple backticks. 
For each audit criterion below, check through the conversation detail provide a detailed assessment with result, including reasons dervied only from the given conversation for the result given.
All evaluation of the compliance strictly based on the conservation content. Don't hallucination. All given Criteria must be audited with output.

Conversation:

{dialog}


Audit Criteria:
``` 
-1. Did the telemarketer ask if the customer is keen to explore how they can benefit from IPPFA's services?
-2. Did the telemarketer mentioned the type of financial services IPPFA or IPP can provide?
-3. Check did the telemarketer propose meeting/zoom session/discussion/meetup with company's consultant?
``` 
Return a list of JSON objects that provide assessments for each of the criteria. Each JSON object should contain the following keys:
*Question: State the criterion being evaluated. eg, "Did the telemarketer ask if the customer is keen to explore how they can benefit from IPPFA's services?"
*Reason: Explain by pointing out specific reason based on the information in the conversation. eg, "The telemarketer did ask if the customer is keen to explore how they can benefit from IPPFA's services."
*Result: Indicate whether the criterion was met with "Pass," "Fail," or "Not Applicable".

Output:

"""

def LLM_audit_2Stage(dialog, file_name, prompt1, prompt2,foldername):
    #run openai

    #run the first prompt 
    #get the last question result
    S1 = openai_run(GPT_MODEL,prompt1)
    audit_result_S1 =S1.replace("json","").replace("```","")
    json_object1 = json.loads(audit_result_S1)
    df1 = pd.DataFrame.from_dict(json_object1) 
    df1.insert(0, 'Stage', 'stage1')

    interest= df1.iloc[5][3]

    if( interest == "Pass"):
        S2 = openai_run(GPT_MODEL,prompt2)
        audit_result_S2 =S2.replace("json","").replace("```","")
        json_object2 = json.loads(audit_result_S2)
        df2 = pd.DataFrame.from_dict(json_object2) 
        df2.insert(0, 'Stage', 'stage2')
        df3 = pd.concat([df1,df2], ignore_index = True)
        audit_result= "stage1\n" + audit_result_S1 +"\n stage2\n"+ audit_result_S2
    else:
        df3= df1
        audit_result= "stage1\n" + audit_result_S1
    print("df3:\n")
    print(df3)
    
    fail_count = df3['Result'].str.contains('Fail').sum()
    dict_pass = { 'Stage': 'Final', 'Question':'Result','Reason':'Result','Result':'Pass'}
    dict_fail = { "Stage": "Final", "Question":"Result","Reason":"Result","Result":"Fail"}
    df_pass = pd.DataFrame([dict_pass])
    df_fail = pd.DataFrame([dict_fail])

    if fail_count>0:
        df4 = pd.concat([df3,df_fail], ignore_index = True)
    else:
        df4 = pd.concat([df3,df_pass], ignore_index = True)


    output_filename= os.path.basename(file_name).split('.')[0]
    output_filename_csv = foldername +"//"+output_filename +".csv"
    transcript_filename_txt = foldername +"//"+output_filename +".txt"
    df4.to_csv(output_filename_csv, index=False) 
    
    f_txt = open(transcript_filename_txt, "a")
    f_txt.write(dialog)
    f_txt.close()
    return audit_result

def select_folder():
   root = tk.Tk()
   root.wm_attributes('-topmost', 1)
   root.withdraw()
   folder_path = filedialog.askdirectory(parent=root)
    
   root.destroy()
   return folder_path


def main():
    st.title("AI Audio Audit App")
    st.write("Upload an audio file and convert it to text.")

    def uploader_callback():
        print('Uploaded file')
         #remove current file
        if(len(file_list)>0):
            for file_name in file_list:
                os.remove("./"+ file_name )
                
    
    # Create a list to store the uploaded file names
    file_list = []
    # Use the file_uploader to allow users to upload files
    uploaded_files = st.file_uploader("Upload Files", on_change=uploader_callback,accept_multiple_files=True)

   
    # Loop through the uploaded files
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # print(uploaded_file)
            # Append the name of the uploaded file to the file_list
            file_list.append(uploaded_file.name)
            save_audio_file(uploaded_file.read(), uploaded_file.name)

    selected_folder_path = st.session_state.get("folder_path", None)
    folder_select_button = st.button("Select Folder")
    if folder_select_button:
      selected_folder_path = select_folder()
      st.session_state.folder_path = selected_folder_path

    if selected_folder_path:
       st.write("Selected folder path:", selected_folder_path)

    # Display the list of uploaded file names
    # st.write("Uploaded Files:")
    # for file_name in file_list:
    #     # Provide a checkbox for each file name to allow users to remove files
    #     remove_file = st.checkbox(file_name)
    #     if remove_file:
    #         # Remove the file name from the file_list if the checkbox is selected
    #         os.remove("./"+ file_name )
    #         file_list.remove(file_name)

    #Submit button is clicked
    submit = st.button("Submit")
    if submit:
        #Create a directory based on the date time
        d = datetime.datetime.now()
        timestamp = "%04d%02d%02d%02d%02d" % (d.year, d.month, d.day, d.hour, d.minute)
        folderWithDate = ".//tmp//" + str(timestamp)
        os.mkdir(folderWithDate)
        #create a log file in the folderwithdate
        f = open(folderWithDate+"//log.txt", "a")
        
        try:
            
            #loop through each files
            for file_name in file_list:
                 st.write(file_name)
                 print("++",file_name)
                 starttime= "%04d%02d%02d%02d%02d" % (d.year, d.month, d.day, d.hour, d.minute)
                 f.write(str(starttime))
                 f.write(file_name)
                             
                 text = speech_to_text(file_name)
                 st.write("Converted Text: " +file_name )
                 st.write(text)
                 
                 st.write("Audit result:")
                 result=LLM_audit_2Stage(text,file_name, prompt1, prompt2, folderWithDate)
                 st.write(result)
    
                 endtime= "%04d%02d%02d%02d%02d" % (d.year, d.month, d.day, d.hour, d.minute)
                 f.write(str(endtime))
                 
            f.close()
        except Exception as e:
             # By this way we can know about the type of error occurring
             print("The error is: ",e)
             f.write("The error is: "+ str(e))
             f.close()
   

if __name__ == "__main__":
    main()
