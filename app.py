from flask import Flask,render_template,request
from Summarize import summarize
app=Flask(__name__)
@app.route("/")
def home():
    return render_template("Input.html",Text="hello")
@app.route("/result",methods=["POST"])
def result():
    form=request.form
    if request.method=='POST':
        text=request.form['Text']
        text=summarize(text)
        print(text)
        return render_template('Output.html',Text=text)

if __name__=="__main__":
    app.run("localhost",1000)