from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time

# Initialize the Flask application
app = Flask(__name__)

INPUT_FILE='http://127.0.0.1:5001/video_feed'
LABELS_FILE='../question0/coco.names'
CONFIG_FILE='../question0/yolov3-tiny.cfg'
WEIGHTS_FILE='../question0/yolov3-tiny.weights'
CONFIDENCE_THRESHOLD=0.3



LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

vs = cv2.VideoCapture(INPUT_FILE)

detections = []

#--------------------------------------------UI--------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

#--------------------------------------------API--------------------------------------------------
## Utilise paramètres GET nomé flux exemple d'appel : /change_flux?flux=http://127.0.0.1:5001/video_feed
@app.route('/change_flux')
def change_flux():
    flux = request.args.get("flux")
    global vs
    vs.release()
    vs = cv2.VideoCapture(flux)
    return flux

@app.route('/video_feed_detection')
def video_feed_detection():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get_detection")
def summary():
    global detections
    return jsonify(detections)

def gen_frames():
    H=None
    W=None
    global detections
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    cnt =0;
    detect=0;
    
    while True:
        cnt+=1
        try:
            (grabbed, image) = vs.read()
        except:
            break

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        if W is None or H is None:
            (H, W) = image.shape[:2]
        layerOutputs = net.forward(ln)


        if cnt%10 == 0:
            detections.append({})
            detections[detect]['timestamp'] = time.time()
            listDetect = []
            
            
            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > CONFIDENCE_THRESHOLD:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                CONFIDENCE_THRESHOLD)

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                j = 0
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    color = [int(c) for c in COLORS[classIDs[i]]]
                    listDetect.append({})
                    listDetect[j]['label'] = LABELS[classIDs[i]]
                    listDetect[j]['x'] = x
                    listDetect[j]['y'] = y
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
                    j+=1
            detections[detect]['list'] = listDetect
            detect += 1
            # return image
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        

    cv2.destroyAllWindows()

    # release the file pointers
    print("[INFO] cleaning up...")
    vs.release()


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5002)
