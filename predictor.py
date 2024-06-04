# this file stores the functions used for prediction
from typing import List, TypedDict
from ultralytics import YOLO
import torch
import cv2
from time import time

class findAttacksRet(TypedDict):
    totalFrames: int
    attackFrames: List[int]

def frameToTime(fps, frame):
    ms = frame / fps * 1000
    minutes = ms // (1000 * 60)
    seconds = (ms / 1000) - (minutes * 60)
    
    return f"{int(minutes)}:{int(seconds):02d}"

def addToWindow(window, windowSize, newElem):
    for i in range(windowSize - 1):
        window[i] = window[i + 1]
    
    window[windowSize - 1] = newElem

def findAttacks(
        videoPath: str, 
        generateVideo: bool = False,
        outputVideoPath: str = "./output/prediction.avi",
        confidence: float = 0.5
    ) -> findAttacksRet:
    """given the path of a video, returns the frameNumber of all spikes.
    Parameters:
        videoPath - path of video of which to find attacks
        generateVideo - generate output video with given 
        outputVideoName - name of output video in same directory of videoPath
        spacing - number of frames in between prediction frames 

    Returns:
        total number of frames processed and the frames on which 
    """

    returnValue:findAttacksRet = {"totalFrames": 0, "attackFrames": []}
    if 0 > confidence or confidence > 1: 
        print("Invalid confidence value", confidence)
        return returnValue

    # load model
    # use gpu if available
    if torch.cuda.device_count() > 0:
        torch.cuda.set_device(0)
    model = YOLO("./best.pt")

    # extract frames from video
    vid = cv2.VideoCapture(videoPath)
    fps = vid.get(cv2.CAP_PROP_FPS)
    height = None
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = None
    outVideo = cv2.VideoWriter();
    
    if (generateVideo):
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        outVideo = cv2.VideoWriter(outputVideoPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # sliding window
    actionWindow = [0] * 10
    attacksInWindow = 0
    lastAttackFrame = -1000

    # progress information
    start = time()
    prev = start
    print("Started")
    for f in range(length):
        ret, frame = vid.read()
        if ret:
            # run the ai model on each frame
            p = model.predict(frame, conf=confidence, classes=[0], verbose=False)[0]

            # output frame to video to be outputted
            if(generateVideo):
                outVideo.write(p.plot())
            
            # add attack if needed
            try:
                summary = p.summary()
                # attack identified in frame
                framePrediction = summary[0]
                if framePrediction['name'] == "Attack":
                    # update attacksInWindow
                    if actionWindow[0] == 0:
                        attacksInWindow += 1
                    addToWindow(actionWindow, 10, 1)
            except:
                # no attacks in frame
                if actionWindow[0] == 1:
                    attacksInWindow -= 1

                addToWindow(actionWindow, 10, 0)
            
            # print progress information
            printTimeSpacing = 15 # number of frames between each progress update
            if f % (fps * printTimeSpacing) == 0 and f != 0: 
                curr = time()
                timeTaken = curr - prev
                processRate = (fps * printTimeSpacing) / timeTaken # in frames per second
                framesLeft = length - f
                secondsRemaining = framesLeft / processRate
                print(
                    f"{f / (fps * 60):.2f} minute(s) processed, {(f / length) * 100:.2f}% done, {int(secondsRemaining // 60)} minutes {int(secondsRemaining % 60)} seconds left"
                )
                prev = curr
                
        else:
            print(f"no frame {f}")

        if attacksInWindow >= 7 and f - lastAttackFrame > fps:
            # attack event registered
            print(f"spike found at {frameToTime(fps, f - 9)}")
            lastAttackFrame = f
            returnValue["attackFrames"].append(f - 9)

    totalTimeTaken = time() - start
    print(f"Analysis complete, time taken: {int(totalTimeTaken // 60)} minutes {int(totalTimeTaken % 60)} seconds ")
    returnValue["totalFrames"] = length
    cv2.destroyAllWindows()
    outVideo.release()

    return returnValue


