import predictor
import cv2
import sys

def main():
    #default values
    videoPath = None
    confidence = 0.5
    outputTextFile = "./output/prediction.txt"
    outputVideoFile = "./output/prediction.avi"
    generateVideo = False
    
    # get command line arguments
    args = sys.argv
    if len(args) == 1:
        print("Not enough arguments")
        print(
            """Usage: main.py <videoPath> [confidence] [outputTextFile] [generateVideo] [generatedVideoName] 
            """)
        return
    if len(args) >= 2:
        videoPath = args[1]
    if len(args) >= 3:
        confidence = float(args[2])
    if len(args) >= 4:
        outputTextFile = args[3]
    if len(args) >= 5:
        generateVideo = bool(args[4])
    if len(args) >= 6:
        outputVideoFile = args[5]

    file = open(outputTextFile, "a")

    # get video fps
    vid = cv2.VideoCapture(videoPath)
    fps = vid.get(cv2.CAP_PROP_FPS)

    # run prediction model
    ret = predictor.findAttacks(videoPath, generateVideo, outputVideoFile, confidence)
    
    for i, p in enumerate(ret['attackFrames']):
        if i % 10 == 0 and i != 0:
            file.write("\n")
        file.write(f"{predictor.frameToTime(fps, p - fps)} ")
        
    file.close()

    


if __name__ == '__main__':
    main()
   
        
    