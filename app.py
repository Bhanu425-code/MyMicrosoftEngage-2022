####### Import Libraries #############
import streamlit as st
import cufflinks as cf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import seaborn as sns
st.set_page_config(layout="wide")
imp=['Customer Segmentation','Price Prediction']

choice1 = st.sidebar.selectbox("Select requirement: ", imp)
####### Load Dataset #####################
def load_dataset(dataset):
    df = pd.read_csv(dataset)
    return df

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def load_prediction_models(model_file):
    load_model = joblib.load(open(os.path.join(model_file), "rb"))
    return load_model

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val==value:
            return key
####### Customer Segmentation ###########
if(choice1=='Customer Segmentation'):


    data = load_dataset("customer_segmentation.csv")
    data.drop(["ID"], axis=1, inplace=True)
    data.drop(["Var_1"], axis=1, inplace=True)

    class_label = {'SUV': 0, 'Sedan': 1, 'Hatchback': 2, 'Micro': 3}

    tasks = ['Data Analysis', 'Prediction']
    choices = st.sidebar.selectbox("Select Task: ", tasks)
    if choices=='Data Analysis':
        st.markdown("## Customer Segmentation")   ## Main Title

################# Scatter Chart Logic #################

        st.sidebar.markdown("### Scatter Chart :")

        ingredients = data.drop(labels=["Segmentation"], axis=1).columns.tolist()

        x_axis = st.sidebar.selectbox("X-Axis", ingredients)
        y_axis = st.sidebar.selectbox("Y-Axis", ingredients, index=1)

        if x_axis and y_axis:
            scatter_fig = data.iplot(kind="scatter", x=x_axis, y=y_axis,
                            mode="markers",
                            categories="Segmentation",
                            asFigure=True, opacity=1.0,
                            xTitle=x_axis.replace("_"," ").capitalize(), yTitle=y_axis.replace("_"," ").capitalize(),
                            title="{} vs {}".format(x_axis.replace("_"," ").capitalize(), y_axis.replace("_"," ").capitalize()),
                            )

########## Bar Chart Logic ##################

        st.sidebar.markdown("### Bar Chart: ")

        avg_data = data.groupby(by=["Segmentation"]).mean()

        bar_axis = st.sidebar.multiselect(label="Bar Chart", options=avg_data.columns.tolist(), default=["Age","Work_Experience"])

        if bar_axis:
            bar_fig = avg_data[bar_axis].iplot(kind="bar",
                                barmode="stack",
                                xTitle="Segmentation",
                                title="Distribution of data for each Segmentation",
                                asFigure=True,
                                opacity=1.0,
                                );
        else:
            bar_fig = avg_data[["Age"]].iplot(kind="bar",
            barmode="stack",
                xTitle="Segmentation",
                title="Distribution of Age for each Segmentation",
                asFigure=True,
                opacity=1.0,
                );

################# Histogram Logic ########################

        st.sidebar.markdown("### Histogram : ")

        hist_axis = st.sidebar.multiselect(label="Histogram", options=ingredients, default=["Profession"])
        bins = st.sidebar.radio(label="Bins :", options=[10,20,30,40,50], index=1)

        if hist_axis:
            hist_fig = data.iplot(kind="hist",
                                    keys=hist_axis,
                                    xTitle="x_axis",
                                    yTitle="y_axis",
                                    bins=bins,
                                    title="Distribution of Columns",
                                    asFigure=True,
                                    opacity=1.0
                                    );
        else:
            hist_fig = data.iplot(kind="hist",
                                    keys=["Graduated"],
                                    xTitle="Graduated",
                                    bins=bins,
                                    title="Distribution of Profession",
                                    asFigure=True,
                                    opacity=1.0
                                    );


#################### Pie Chart Logic ##################################

        w_cnt = data.groupby(by=["Segmentation"]).count()[['Gender']].rename(columns={"Gender":"Count"}).reset_index()

        pie_fig = w_cnt.iplot(kind="pie", labels="Segmentation", values="Count",
                            title="Segmentation",
                            hole=0.4,
                            asFigure=True)


##################### Layout Application ##################

        container1 = st.container()
        col1, col2 = st.columns(2)

        with container1:
            with col1:
                scatter_fig
            with col2:
                bar_fig


        container2 = st.container()
        col3, col4 = st.columns(2)

        with container2:
            with col3:
                hist_fig
            with col4:
                pie_fig

###################### Prediction ############################3
    if choices=='Prediction':
        Gender_label = {'Male': 0, 'Female': 1}
        Ever_Married_label = {'Yes': 0, 'No': 1}
        Spending_Score_label={'High': 0, 'Average': 1, 'Low': 2}
        Graduated_label = {'Yes': 0, 'No': 1}
        Profession_label = {'Artist': 0, 'Healthcare': 1,'Entertainment':2,'Docter':3,'Engineer':4,'Executive':5,'Laywer':6,'Marketing':7,'Homemaker':8 }

        st.subheader("Lets start with ML")
        Gender = st.selectbox("Select Gender: ", tuple(Gender_label.keys()))
        Ever_Married = st.radio("Marital Status",('Yes', 'No'))
        Work_Experience = st.number_input("Enter your Work Experience: ", 0, 100)
        Family_Size =st.slider('Family Size', 0, 10)
        Spending_Score = st.selectbox("Enter Spending Score: ", tuple(Spending_Score_label.keys()))
        Age = st.number_input("Enter your Age: ", 0, 100)
        Graduated = st.radio("Graduated",('Yes', 'No'))
        Profession = st.selectbox("Select Profession: ", tuple(Profession_label.keys()))
        k_Gender = get_value(Gender,Gender_label)
        k_Ever_Married=get_value(Ever_Married,Ever_Married_label)
        k_Spending_Score = get_value(Spending_Score, Spending_Score_label)
        k_Graduated = get_value(Graduated, Graduated_label)
        k_Profession = get_value(Profession,Profession_label)
        sample_data = [k_Gender, k_Spending_Score, k_Graduated, k_Profession,k_Ever_Married ,Work_Experience,Family_Size,Age ]
        prep_data = np.array(sample_data).reshape(1,-1)
        if st.button('Evaluate'):
            pred = load_prediction_models("customer_seg_descision_tree_model.pkl")
            y_pred = pred.predict(prep_data)
            final_result = get_key(y_pred, class_label)
            st.success(final_result)

############### Price Prediction #####################333
if(choice1=='Price Prediction'):
    data1 = load_dataset("price_prediction.csv")
    tasks = ['Data Analysis', 'Prediction']
    choices = st.sidebar.selectbox("Select Task: ", tasks)
    if choices=='Data Analysis':
        st.markdown("## Price Prediction")   ## Main Title
        st.sidebar.markdown("### Select Axes Here")
        ################# Scatter Chart Logic #################
        st.sidebar.markdown("### Scatter Chart :")

        ingredients1 = data1.drop(labels=["price"], axis=1).columns.tolist()

        x_axis = st.sidebar.selectbox("X-Axis", ingredients1)
        y_axis = st.sidebar.selectbox("Y-Axis", ingredients1, index=1)

        if x_axis and y_axis:
            scatter_fig = data1.iplot(kind="scatter", x=x_axis, y=y_axis,
                        mode="markers",
                        categories="body-style",
                        asFigure=True, opacity=1.0,
                        xTitle=x_axis.replace("_"," ").capitalize(), yTitle=y_axis.replace("_"," ").capitalize(),
                        title="{} vs {}".format(x_axis.replace("_"," ").capitalize(), y_axis.replace("_"," ").capitalize()),
                        )
        ########## Bar Chart Logic ##################

        st.sidebar.markdown("### Bar Chart: ")

        avg_data = data1.groupby(by=["make"]).mean()

        bar_axis = st.sidebar.multiselect(label="Bar Chart", options=avg_data.columns.tolist(), default=["length","width"])

        if bar_axis:
                bar_fig = avg_data[bar_axis].iplot(kind="bar",
                                    barmode="stack",
                                    xTitle="make",
                                    title="Distribution of data for each make",
                                    asFigure=True,
                                    opacity=1.0,
                                    );
        else:
                bar_fig = avg_data[['compression-ratio']].iplot(kind="bar",
                barmode="stack",
                    xTitle="compression_ratio",
                    title="Distribution of data for each make ",
                    asFigure=True,
                    opacity=1.0,
                    );
        ################# Histogram Logic ########################

        st.sidebar.markdown("### Histogram : ")

        hist_axis = st.sidebar.multiselect(label="Histogram", options=ingredients1, default=["aspiration"])
        bins = st.sidebar.radio(label="Bins :", options=[10,20,30,40,50], index=1)

        if hist_axis:
            hist_fig = data1.iplot(kind="hist",
                                    keys=hist_axis,
                                    xTitle="aspiration",
                                    bins=bins,
                                    title="Distribution of Columns",
                                    asFigure=True,
                                    opacity=1.0
                                    );
        else:
            hist_fig = data1.iplot(kind="hist",
                                    keys=["fuel-type"],
                                    xTitle="fuel-type",
                                    bins=bins,
                                    title="Distribution of fuel-type",
                                    asFigure=True,
                                    opacity=1.0
                                    );
#################### Pie Chart Logic ##################################

        w_cnt = data1.groupby(by=["body-style"]).count()[['fuel-type']].rename(columns={"fuel-type":"Count"}).reset_index()

        pie_fig = w_cnt.iplot(kind="pie", labels="body-style", values="Count",
                            title="body-style",
                            hole=0.4,
                            asFigure=True)
######################## Application layout ########################
        container1 = st.container()
        col1, col2 = st.columns(2)

        with container1:
            with col1:
                scatter_fig
            with col2:
                bar_fig


        container2 = st.container()
        col3, col4 = st.columns(2)

        with container2:
            with col3:
                hist_fig
            with col4:
                pie_fig

############### Prediction ###############################
    if choices=='Prediction':
        fuel_type_label = {'gas': 0, 'diesel': 1}
        num_of_doors_label = {'two': 0, 'four': 1}
        make_label={'alfa-romero':0, 'audi, bmw':1, 'chevrolet':2, 'dodge':3, 'honda':4,
                           'isuzu':5, 'jaguar':6, 'mazda':7, 'mercedes-benz':8, 'mercury':9,
                           'mitsubishi':10, 'nissan':11, 'peugot':12, 'plymouth':13, 'porsche':14,
                           'renault':15, 'saab':16, 'subaru':17, 'toyota':18, 'volkswagen':19, 'volvo':20}
        drive_wheels_label={"4wd":0, "fwd":1, "rwd":2}
        aspiration_label = {'std': 0, 'turbo': 1}
        body_style_label = {' hardtop': 0, 'wagon': 1,'sedan':2,'hatchback':3,'convertible':4}
        engine_location_label = {'front': 0, 'rear': 1}
        num_of_cylinders_label = {'two': 0, 'three': 1,'four':2,'five':3,'six':4,'eight':4,'twelve':5}
        engine_type_label={"dohc":0, "dohcv":1, "l":2, "ohc":3, "ohcf":4, "ohcv":5, "rotor":6}
        fuel_system_label={"1bbl":0, "2bbl":1, "4bbl":2, "idi":3, "mfi":4, "mpfi":5, "spdi":6, "spfi":7}


        st.subheader("Lets start with ML")
        fuel_type = st.radio("Select Fuel Type: ", tuple(fuel_type_label.keys()))
        make = st.selectbox("Select Make", tuple(make_label.keys()))
        drive_wheels=st.selectbox("Select Drive Wheels", tuple(drive_wheels_label.keys()))
        num_of_doors = st.radio("Enter Number of doors: ", tuple(num_of_doors_label.keys()))
        aspiration = st.selectbox("Select Aspiration: ", tuple(aspiration_label.keys()))
        body_style = st.selectbox("Select Body Style: ", tuple(body_style_label.keys()))
        engine_location= st.radio("Select Engine location: ", tuple(engine_location_label.keys()))
        num_of_cylinders= st.selectbox("Select Number of cylinders: ", tuple(num_of_cylinders_label.keys()))
        engine_type=st.selectbox("Select Engine type", tuple(engine_type_label.keys()))
        fuel_system=st.selectbox("Select Fuel System", tuple(fuel_system_label.keys()))


        engine_size = st.number_input("Engine size: ", 61, 326)
        normalized_losses = st.number_input(" Normalized Losses : ", 65, 256)
        horsepower = st.number_input("Horsepower: ", 48, 288)
        wheel_base=st.number_input("Wheel Base: ", 86.6, 120.9)
        compression_ratio = st.number_input("Compression-ratio : ", 7, 23)
        stroke = st.number_input("Stroke value: ", 2.07, 4.17)
        city_mpg = st.number_input("City-mpg : ", 13, 49)
        highway_mpg = st.number_input("Highway-mpg : ", 16, 54)

        symboling =st.slider('Symboling', -3, 3)
        length =st.slider('Length', 141.1, 208.1)
        width =st.slider('Width', 60.3, 72.3)
        height =st.slider('Height', 47.8, 59.8)
        curb_weight =st.slider('Curb weight', 1488, 4066)
        bore =st.slider('Bore', 2.54, 3.94)
        peak_rpm =st.slider('Peak-rpm',4150, 6600)

        k_fuel_type = get_value(fuel_type,fuel_type_label)
        k_make=get_value(make,make_label)
        k_drive_wheels = get_value(drive_wheels, drive_wheels_label)
        k_num_of_doors = get_value(num_of_doors, num_of_doors_label)
        k_aspiration = get_value(aspiration, aspiration_label)
        k_body_style = get_value(body_style,body_style_label)
        k_engine_location = get_value(engine_location,engine_location_label)
        k_num_of_cylinders = get_value(num_of_cylinders,num_of_cylinders_label)
        k_engine_type = get_value(engine_type,engine_type_label)
        k_fuel_system = get_value(fuel_system,fuel_system_label)

        sample_data = [k_fuel_type,k_make,k_drive_wheels,k_num_of_doors,k_aspiration ,k_body_style,k_engine_location,
        k_num_of_cylinders,k_engine_type,k_fuel_system,normalized_losses,wheel_base,stroke,city_mpg,highway_mpg,
        symboling,curb_weight,bore,peak_rpm,engine_size,horsepower,compression_ratio ,length,width,height ]
        prep_data = np.array(sample_data).reshape(1,-1)
        if st.button('Evaluate'):
            pred = load_prediction_models("regression_model_price_pred.pkl")
            y_pred = pred.predict(prep_data)
            st.success(y_pred)
