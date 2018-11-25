#quick and dirty script to extract images from a video, then pass them to microsofts face api to analyze images for emotions
#results are saved to a csv file for further analysis
import cv2
import math
import requests
import csv
import pandas as pd
import os

videoFile = "videos/video4.mp4"
imagesFolder = "images/rendered"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate

#could use as counter if the program takes long, for now just show how many frames the video contains
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
counter=0
filenamearray=[]
framenamearray=[]
finaloutput=[]

print('rendering movie to still images, frames to process: ' + str(video_length))        
# Replace 'KEY_1' with your subscription key as a string, key seems to be tied to uri_base?
subscription_key = 'f0c84016e93a43ce9a7ade8c3d44b381'
uri_base = 'https://westcentralus.api.cognitive.microsoft.com'
headers = {
     'Content-Type': 'application/octet-stream',
     'Ocp-Apim-Subscription-Key': subscription_key,
}
# Request parameters 
# The detection options for MCS Face API check MCS face api 
params = {
    'returnFaceId': 'false',
     'returnFaceAttributes': 'emotion',
}
# route to the face api
path_to_face_api = '/face/v1.0/detect'

#save out still images from video
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    #if (frameId % math.floor(frameRate) == 0):
    #increase number of frames to process for test from frameRate to 6
    if (frameId % math.floor(frameRate) == 0):
        filename = imagesFolder + "/image_" +  str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, frame)
        #need to save image filenames and send them to ml function to process
        filenamearray.append(filename)
        framenamearray.append(frameId)        
cap.release()

#loop through list of images
for imagetoprocess in filenamearray:
    # open jpg file as binary file data for intake by the MCS api
    with open(imagetoprocess, 'rb') as f:
        img_data = f.read()
    try:
        print("parsing file: " + str(counter+1) + " out of: " + str(len(framenamearray)))
        # Execute the api call as a POST request. 
        # Note: mcs face api only returns 1 analysis at time
        response = requests.post(uri_base + path_to_face_api,
                                data=img_data, 
                                headers=headers,
                                params=params)
        # json() is a method from the request library that converts 
        # the json reponse to a python friendly data structure
        parsed = response.json()
            
        # convoluted way of saving....
        anger = float(parsed[0]['faceAttributes']['emotion']['anger'])
        contempt = float(parsed[0]['faceAttributes']['emotion']['contempt'])
        disgust = float(parsed[0]['faceAttributes']['emotion']['disgust'])
        fear = float(parsed[0]['faceAttributes']['emotion']['fear'])
        happiness = float(parsed[0]['faceAttributes']['emotion']['happiness'])
        neutral = float(parsed[0]['faceAttributes']['emotion']['neutral'])
        sadness = float(parsed[0]['faceAttributes']['emotion']['sadness'])
        surprise = float(parsed[0]['faceAttributes']['emotion']['surprise'])        
        finaloutput.append([])
        finaloutput[counter].append(counter)
        finaloutput[counter].append(anger)
        finaloutput[counter].append(contempt)
        finaloutput[counter].append(disgust)
        finaloutput[counter].append(fear)
        finaloutput[counter].append(happiness)
        finaloutput[counter].append(neutral)
        finaloutput[counter].append(sadness)
        finaloutput[counter].append(surprise)
        counter = counter+1

    except Exception as e:
        print('Error:')
        print(e)
        print(parsed)
        #fill in frames that drop with 0
        finaloutput.append([])
        finaloutput[counter].append(counter)
        finaloutput[counter].append(0)
        finaloutput[counter].append(0)
        finaloutput[counter].append(0)
        finaloutput[counter].append(0)
        finaloutput[counter].append(0)
        finaloutput[counter].append(0)
        finaloutput[counter].append(0)
        finaloutput[counter].append(0)
        counter=counter+1

#write out data to csv file
csvfile = "csv_output.csv"
df = pd.DataFrame(finaloutput)
df.columns = ["framenumber", "anger", "contempt", "disgust", "fear","happiness","neutral", "sadness", "surprise"]
df.to_csv(csvfile, sep=',', encoding='utf-8', index=False)

#clean up temporary images
for imagetoprocess in filenamearray:
    os.remove(imagetoprocess)
