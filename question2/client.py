from flask import Flask, render_template, Response, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

#--------------------------------------------UI--------------------------------------------------
@app.route('/')
def index():
    return render_template('client.html')

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5003)