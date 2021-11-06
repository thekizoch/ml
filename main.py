from flask import Flask, render_template, request

# Configure application
app = Flask(__name__)

@app.route("/")
def get_index():
    return render_template('index.html')

@app.route('/<model>',methods = ['GET','POST'])
def single_page(model):
    return render_template(f'{model}.html')


# BELOW IN DEV - a natural language model:
# the load step of model takes too much time - try it as a cloud function, or a smaller model
# where/can i have it loaded to be ready to be called? first get experience
# in smaller models on Flask

@app.route('/model_bert_qa',methods = ['GET','POST'])
def model_bert_qa():
    return render_template('model_bert_qa.html')

@app.route('/model_bert_qa_result', methods=['GET', 'POST'])
def model_bert_qa_result():
    if request.method == 'POST':

        user_input = request.form['user_input']

        # questions = [user_input]

        return render_template('model_bert_qa_result.html', \
            user_input=user_input)


if __name__ == "__main__":
    #app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    app.run(debug=True)
