from flask import Flask, render_template,request
from tumor import predict_tumor
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
MODAL_path = 'model/model.pth'
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/detect', methods=['GET','POST'])
def detect():
    if request.method=='POST':
         print("posted")
         image_file = request.files['file'] 
         filename = secure_filename(image_file.filename)
         path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
         image_file.save(path)
         result = predict_tumor(path,MODAL_path)
         print(result)
         return { 'tumorFound' : result}
    
    return render_template('detect.html')

if __name__ == "__main__":
        app.run(debug=True)