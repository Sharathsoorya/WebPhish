from flask import Flask, request, render_template, flash
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')
from features import FeatureExtraction


app = Flask(__name__)
# app.secret_key = "123abc$#@!"


with open('model','rb') as f:
    rf_model = pickle.load(f)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template("index.html")
@app.route("/result",methods=['POST','GET'])
def result():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30)


        y_pred =rf_model.predict(x)

        if y_pred == -1:
            return render_template("unsafe.html")
        else:
            return render_template("safe.html")



if __name__ == "__main__":
    app.run(debug = False, host = '0.0.0.0')
#%%
