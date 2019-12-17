from flask import Flask,request,render_template
import numpy as np
import cv2
import tensorflow as tf
model = tf.keras.models.load_model('Hieubui.h5')

food = ['Bread','Dairy product','Dessert','Egg','Fried food','Meat','Noodles','Rice','Seafood','Soup','Vegetable']
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result',methods = ['POST','GET'])
def result():
    if request.method == 'POST':
        result = request.form['filename']
        img = cv2.resize(cv2.cvtColor(cv2.imread("test/"+result),cv2.COLOR_BGR2RGB),(224,224)).astype("float32")
        test_list = np.expand_dims(img,axis = 0)
        pred = model.predict(test_list)
        res = np.argmax(pred)
        percent = round(pred[0][res]*100,2)
        return render_template("result.html",result = food[res],percent = percent)
if __name__ == "__main__":
    app.run(debug=True)
