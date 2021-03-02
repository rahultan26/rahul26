import cv2
import numpy as np


#loading the coco dataset where YOLO was trained

labelspath= "cocon/coco.names"


#LABELS is used to extract different labels from coco data set (e.g. car,person)
LABELS = open(labelspath).read().strip().split("\n")

#initialises a list of colors to represnt each possible class
#randomly selects a number which is further used to assign colors
np.random.seed(42)
COLORS= np.random.randint(0,255,size=(len(LABELS),3),dtype="uint8")


#the path to our YOLO weight and model config
#weight= neural network training (wixi+bias)
weightspath="coco weights/yolov3.weights"
configpath="yoloc/yolov3.cfg"

#loading YOLO
print("loading YOLO......")
#darknet is a framework in which YOLO is written

net=cv2.dnn.readNetFromDarknet(configpath,weightspath)

#YOLOv3 has 3 output layers
#getLayerNames(): Get the name of all layers of the network.
#getUnconnectedOutLayers(): Get the index of the output layers.
ln=net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs=cv2.VideoCapture(r"C:\Users\RAHUL PC\Desktop\vehicle counter\car.mp4")
writer = None
(W, H) = (None, None)

#loop over the frames from the video file

while True:
    #reading the next frame
    #grab:for grabbing th next frame ..returns true..

    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]


    #constructing a blob
    # PASS TO YOLO
    blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
    net.setInput(blob)
    layerOutputs=net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively

    boxes=[]
    confidences=[]
    classIDs=[]

    #loop over output
    for output in layerOutputs:
        #loop over each of the detections
        for detection in output:
            #extract the classIDs and confidence
            scores=detection[5:]
            classID=np.argmax(scores)
            confidence=scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > (0.5):
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    #applying NMS to bounding boxes
    idxs=cv2.dnn.NMSBoxes(boxes,confidences,.5,.3)

    #ENSURING AT LEAST ONE DETECTION EXISTS

    if len(idxs)>0:
        #loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                       confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("output",frame)
    key = cv2.waitKey(1)
    if key == 32:
        break


vs.release()
cv2.destroyAllWindows()