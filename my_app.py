from flask import Flask, render_template,request
import pickle
from flask_cors import CORS,cross_origin
import pandas as pd
from flask_cors import CORS
app=Flask(__name__)
cors=CORS(app)
df=pd.read_csv('car_price_prediction.csv')
with open('models.pkl', 'rb') as f:
    model = pickle.load(f)

with open('pipeline.pkl', 'rb') as f:
    full_pipeline = pickle.load(f)
@app.route('/',methods=['GET','POST'])
def index():
    Manufacturer_companies=sorted(df['Manufacturer'].unique())
    Car_categories=sorted(df['Category'].unique())
    model_in_year=sorted(df['Prod. year'].unique())
    fuel_engine_type=sorted(df['Fuel type'].unique())
    # engine_volume_in=sorted(df['Turbo'].unique())
    car_gear_type=sorted(df['Gear box type'].unique())
    Car_color=sorted(df['Color'].unique())
    Car_drive_wheels=sorted(df['Drive wheels'].unique())
    Car_control_wheel=sorted(df['Wheel'].unique())
    Car_doors=sorted(df['Doors'].unique())

    return render_template('index.html' , Manufacturer_companies=Manufacturer_companies ,Car_categories=Car_categories ,model_in_year=model_in_year,fuel_engine_type=fuel_engine_type,car_gear_type=car_gear_type,
                           Car_color=Car_color,Car_drive_wheels=Car_drive_wheels,Car_control_wheel=Car_control_wheel, Car_doors= Car_doors )
@app.route('/predict' , methods=['POST'])
@cross_origin()
def predict():

    Manufacturer=request.form.get('Manufacturer')
    Category = request.form.get('Category')
    Mileage = request.form.get('Mileage')
    model_in_year = request.form.get('Prod. year')
    fuel_engine_type = request.form.get('Fuel type')

    engine_volume_in = request.form.get('Turbo')
    car_gear_type = request.form.get('Gear box type')
    Car_color = request.form.get('Color')
    Car_drive_wheels= request.form.get('Drive wheels')
    Car_control_wheel = request.form.get('Wheel')
    Car_doors = request.form.get('Doors')
    Car_lavy = request.form.get('Levy')
    Car_airbags = request.form.get('Airbags')
    Car_total_cylinder = request.form.get('Cylinders')
    Car_engine = request.form.get('Engine volume')


    input_data = pd.DataFrame([[Manufacturer, Category, Mileage, model_in_year, fuel_engine_type,
                            Car_engine, engine_volume_in, Car_total_cylinder, car_gear_type,
                            Car_airbags, Car_color, Car_control_wheel, Car_drive_wheels, Car_doors,
                            Car_lavy]],
                          columns=['Manufacturer', 'Category', 'Mileage', 'Prod. year', 'Fuel type',
                                   'Engine volume', 'Turbo', 'Cylinders', 'Gear box type',
                                   'Airbags', 'Color', 'Wheel', 'Drive wheels', 'Doors', 'Levy'])

    processed_input = full_pipeline.transform(input_data)



    prediction = model.predict(processed_input)

    return f"Predicted Car Price: {prediction[0]:,.2f}"
    # print(model.predict([[Manufacturer,Category,Mileage,model_in_year,fuel_engine_type,Car_engine,engine_volume_in,Car_total_cylinder,car_gear_type,Car_airbags
    #                       ,Car_color,Car_control_wheel,Car_drive_wheels,Car_doors,Car_lavy]]))
    # return ""


if __name__=='__main__':
    app.run(debug=True)


