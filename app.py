from os import P_PID
import flask
import pickle
import pandas as pd

app = flask.Flask(__name__, template_folder='Templates')



# Use pickle to load in the pre-trained model.
with open(f'model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        mean = flask.request.form['mean']
        categ_0 = flask.request.form['categ_0']
        categ_1 = flask.request.form['categ_1']
        categ_2 = flask.request.form['categ_2']
        categ_3 = flask.request.form['categ_3']
        categ_4 = flask.request.form['categ_4']
        input_variables = pd.DataFrame([[mean, categ_0, categ_1, categ_2, categ_3,categ_4]],
                                       columns=['mean', 'categ_0', 'categ_1', 'categ_2','categ_3','categ_4'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('index.html',
                                     original_input={'Selling Price':mean,
                                                     'Marked Price':categ_0},
                                     result=prediction,
                                     )



if __name__ == "__main__":
    app.run(debug=True)