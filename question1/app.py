from flask import Flask, render_template, Response
import cv2

# Initialize the Flask application
app = Flask(__name__)

INPUT_FILE='../question0/voitures.mp4'

vs = cv2.VideoCapture(INPUT_FILE)

#--------------------------------------------UI--------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

#--------------------------------------------API--------------------------------------------------
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():  
    while True:
        try:
            (grabbed, frame) = vs.read()
        except:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


if __name__ == '__main__':
    app.run()
