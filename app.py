from flask import Flask,render_template,url_for,request
from flask_material import Material


# EDA PKg
import pandas as pd
import numpy as np

# ML Pkg
from sklearn.externals import joblib


app = Flask(__name__)
#Material(app)


@app.route('/')
def homepage():
    return render_template("homepage.html")

@app.route('/index') #predict button
def index():
    return render_template("index.html")

@app.route('/contact.html')
def contact():
    return render_template("contact.html")


@app.route('/preview')
def preview():
    df = pd.read_csv('data/clean_data.csv')
    return render_template("preview.html", df_view=df)



@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    if request.method == "POST":
      
        percentage10 = request.form.get('10percentage')
        percentage12 = request.form.get('12percentage')
        collegeGPA = request.form.get('collegeGPA')
        English = request.form.get('English')
        Logical = request.form.get('Logical')
        Quant = request.form.get('Quant')
        Domain = request.form.get('Domain')
        ComputerProgramming = request.form.get('CP')
        ComputerScience = request.form.get('CS')
        conscientiousness = request.form.get('conscientiousness')
        agreeableness = request.form.get('agreeableness')
        extraversion = request.form.get('extraversion')
        nueroticism = request.form.get('nueroticism')
        openess_to_experience = request.form.get('openess_to_experience')
        GradAge12 = request.form.get('12GradAge')
        GradAge = request.form.get('GradAge')
        Gender = request.form.get('Gender')
        board12 = request.form.get('12board')
        Degree = request.form.get('degree')
        Specialization = request.form.get('Specialization')

        ###one hot encoding###

        if Gender=='male':
          Gender_m=1
          Gender_f=0
        else:
          Gender_m=0
          Gender_f=1

        if board12=='cbse':
          board12_cbse=1
          board12_icse=0
          board12_na=0
          board12_state=0
        elif board12=='icse':
          board12_cbse=0
          board12_icse=1
          board12_na=0
          board12_state=0
        elif board12=='n/a':
          board12_cbse=0
          board12_icse=0
          board12_na=1
          board12_state=0;
        elif board12=='state':
          board12_cbse=0
          board12_icse=0
          board12_na=0
          board12_state=1;

        if Degree=='BE':
          Degree_BE=1
          Degree_MSc=0
          Degree_ME=0
          Degree_MCA=0;
        elif Degree=='MSc':
          Degree_BE=0
          Degree_MSc=1
          Degree_ME=0
          Degree_MCA=0;
        elif Degree=='ME':
          Degree_BE=0
          Degree_MSc=0
          Degree_ME=1
          Degree_MCA=0;
        elif Degree=='MCA':
          Degree_BE=0
          Degree_MSc=0
          Degree_ME=0
          Degree_MCA=1;

        if Specialization=='CS':
          Specialization_CS=1
          Specialization_EC=0
          Specialization_EL=0 
          Specialization_ME=0
          Specialization_other=0;
        elif Specialization=='EC':
          Specialization_CS=0
          Specialization_EC=1
          Specialization_EL=0 
          Specialization_ME=0
          Specialization_other=0;
        elif Specialization=='EL':
          Specialization_CS=0
          Specialization_EC=0
          Specialization_EL=1 
          Specialization_ME=0
          Specialization_other=0;
        elif Specialization=='ME':
          Specialization_CS=0
          Specialization_EC=0
          Specialization_EL=0
          Specialization_ME=1
          Specialization_other=0;
        elif Specialization=='other':
          Specialization_CS=0
          Specialization_EC=0
          Specialization_EL=0 
          Specialization_ME=0
          Specialization_other=1;


        sample_data = [percentage10,percentage12,collegeGPA,English,Logical,Quant
        ,Domain,ComputerProgramming,ComputerScience,conscientiousness,agreeableness,
        extraversion,nueroticism,openess_to_experience,GradAge12,GradAge,
        Gender_f, Gender_m,board12_cbse, board12_icse, board12_na, board12_state,
        Degree_BE, Degree_MSc, Degree_ME,
        Degree_MCA, Specialization_CS, Specialization_EC,Specialization_EL, Specialization_ME, Specialization_other
        ]
        print(sample_data)
        print(len(sample_data))
        # Unicode to float
        ###prediction###
        
        clean_data = [float(i) for i in sample_data]
        # Reshape the Data as a Sample not Individual Features
        
        ex1 = np.array(clean_data).reshape(1, -1)
        # loading the model
        rf_model = joblib.load('data/model_rf_compressed')
        # applying input to model
        results = rf_model.predict(ex1)
         #fetchign results of prediction
        result = results.tolist()
        result_prediction=result[0]
        result_prediction_l=round(int(result[0]),-3)
        result_prediction_u=round(int(result[0]),-3)+1000
         
        print((result_prediction_l,result_prediction_u))
    
    #return render_template("index.html",result_prediction_l=result_prediction_l,
        #result_prediction_u=result_prediction_u)
      #result_prediction='hello'
        return render_template("index.html",result_prediction_l=result_prediction_l,result_prediction_u=result_prediction_u)
     

if __name__ == '__main__':
    app.run(debug=True)
