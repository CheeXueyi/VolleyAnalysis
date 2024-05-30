# this file stores the functions used for prediction
from typing import List, TypedDict
from ultralytics import YOLO
import torch
import cv2

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
        generateVideo: bool = True,
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
    
    actionWindow = [0] * 10
    attacksInWindow = 0
    lastAttackFrame = -1000
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

            if f % (fps * 60) == 0: print(f"{frameToTime(fps, f)} processed")
        else:
            print(f"no frame {f}")

        if attacksInWindow >= 7 and f - lastAttackFrame > fps:
            # attack event registered
            print(f"spike found at {frameToTime(fps, f - 9)}")
            lastAttackFrame = f
            returnValue["attackFrames"].append(f - 9)

    print("Done")
    returnValue["totalFrames"] = length
    cv2.destroyAllWindows()
    outVideo.release()

    return returnValue


