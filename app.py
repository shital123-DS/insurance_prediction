from flask import Flask, render_template,request
from artifacts.utils import insurance

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('insurance.html')

@app.route('/predict', methods=["POST"])
def insurance_predict():
    data = request.form
    insurance_obj = insurance(data)
    result = insurance_obj.predict()

    return render_template('insurance.html', pred=result)

if __name__ == "__main__":
    app.run(debug=True)