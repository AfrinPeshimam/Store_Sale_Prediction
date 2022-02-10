from flask import Flask,request,render_template,jsonify
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


app= Flask(__name__)

model= joblib.load('xgb_model.pickle')
oe= joblib.load('ordinal_encoder.pickle')
cat_col= ['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']

@app.route('/')
def homePage():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method=="POST":
        Item_Weight= float(request.form['Item_Weight'])
        Item_Fat_Content= str(request.form['Item_Fat_Content'])
        Item_Visibility= float(request.form['Item_Visibility'])
        Item_Type= str(request.form['Item_Type'])
        Item_MRP= float(request.form['Item_MRP'])
        Outlet_Identifier= str(request.form['Outlet_Identifier'])
        Outlet_Establishment_Year= int(request.form['Outlet_Establishment_Year'])
        Outlet_Size= str(request.form['Outlet_Size'])
        Outlet_Location_Type= str(request.form['Outlet_Location_Type'])
        Outlet_Type= str(request.form['Outlet_Type'])



        df= pd.DataFrame({
            'Item_Weight': [Item_Weight],
            'Item_Fat_Content': [Item_Fat_Content], 
            'Item_Visibility': [Item_Visibility], 
            'Item_Type': [Item_Type], 
            'Item_MRP': [Item_MRP],
            'Outlet_Identifier': [Outlet_Identifier],
            'Outlet_Establishment_Year': [Outlet_Establishment_Year],
            'Outlet_Size': [Outlet_Size],
            'Outlet_Location_Type': [Outlet_Location_Type],
            'Outlet_Type': [Outlet_Type],
        })
        df[cat_col]= oe.transform(df[cat_col])
        preds= model.predict(df)[0]


    return render_template('results.html', prediction=preds)

if __name__ == '__main__':
    app.run(debug=True)
