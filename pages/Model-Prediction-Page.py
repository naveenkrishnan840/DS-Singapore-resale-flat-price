import numpy as np
import streamlit as st
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
import json
import datetime
import calendar
from feature_engine.creation import CyclicalFeatures
import mlflow
import dagshub

st.markdown("<h2 style='font-family: san-serif; color: red; text-align: center;'>"
            "Model Prediction</h2>", unsafe_allow_html=True)

task = st.selectbox(label="Select ML Task for prediction", options=["Select", "Prediction"])

if task != "Select":
    if task == "Prediction":
        with (st.container(border=1)):
            @st.cache_data
            def model():
                dagshub.init(repo_owner='naveenkrishnan840', repo_name='DS-Singapore-resale-flat-price',
                             mlflow=True)
                return mlflow.sklearn.load_model(model_uri="models:/Regression-Task/4")
            regression_model = model()


            def search_block():
                set_block = False
                with open("./block.json") as file:
                    block_json = json.load(file)
                    for block_name, block_value in block_json.items():
                        if st.session_state.block_key == block_name:
                            st.session_state.set_block = block_value
                            set_block = True
                            break
                    if not set_block:
                        st.session_state.block_key = ""
                        st.session_state.set_block = ""

            st.success(f"Based On Model Training We should take classification Task Model Version 9. "
                       f"Version 9 is very good accuracy for both training & testing. Accuracy Score is "
                       f"{round(regression_model.best_score_.tolist() * 100)} %")
            st.divider()
            month_col = st.columns(1)
            with month_col[0]:
                with st.expander('Month'):
                    this_year = datetime.date.today().year
                    this_month = datetime.date.today().month
                    flat_year = st.selectbox('', range(this_year, this_year - 8, -1), label_visibility="hidden")
                    month_abbr = calendar.month_abbr[1:]
                    flat_month_str = st.radio('', month_abbr, index=this_month - 1, horizontal=True)
                    flat_month = month_abbr.index(flat_month_str) + 1
                st.text(f'{flat_year} {flat_month_str}')
            st.divider()
            flat_block = st.columns(2)
            with flat_block[0]:
                flat_type = st.selectbox(label="Flat Type", options=["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM",
                                                                     "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"])
            with flat_block[1]:
                block = st.text_input(label="Block", on_change=search_block, key="block_key")
            st.divider()
            storey_floor_area_sqm = st.columns(2)
            with storey_floor_area_sqm[0]:
                storey_range = st.selectbox(label="Storey Range", options=["01 TO 03", "04 TO 06", "07 TO 09",
                                                                           "10 TO 12", "13 TO 15", "16 TO 18",
                                                                           "19 TO 21", "22 TO 24", "25 TO 27",
                                                                           "28 TO 30", "31 TO 33", "34 TO 36",
                                                                           "37 TO 39", "40 TO 42", "43 TO 45",
                                                                           "46 TO 48", "49 TO 51"])
            with storey_floor_area_sqm[1]:
                floor_area_sqm = st.number_input(label="Floor Area Sqm")
            st.divider()
            flat_model_lease_commence_date = st.columns(2)
            with flat_model_lease_commence_date[0]:
                flat_model = st.selectbox(label="Floor Model", options=["Improved", "New Generation", "DBSS",
                                                                         "Standard", "Apartment", "Simplified",
                                                                         "Model A", "Premium Apartment",
                                                                         "Adjoined flat", "Model A-Maisonette",
                                                                         "Maisonette", "Type S1", "Type S2",
                                                                         "Model A2", "Terrace", "Improved-Maisonette",
                                                                         "Premium Maisonette", "Multi Generation",
                                                                         "Premium Apartment Loft", "2-room", "3Gen"])
            with flat_model_lease_commence_date[1]:
                with st.expander('Lease Commence Date'):
                    pre_year = datetime.date.today().year - 4
                    lease_commence_date = st.selectbox('', range(pre_year, 1979, -1))
                st.text(lease_commence_date)
            rem_lease_col = st.columns(2)
            with rem_lease_col[0]:
                year_rem_lease = st.select_slider("Select a Year Remaining Lease", options=list(range(41, 98)))
            with rem_lease_col[1]:
                month_rem_lease = st.select_slider("Select a Month Remaining Lease", options=list(range(0, 13)))
            st.divider()
            click_btn = st.button("Submit", use_container_width=True)
            if click_btn and (flat_year and flat_month and flat_type and st.session_state.set_block and storey_range
                              and (floor_area_sqm or not(floor_area_sqm == 0.0)) and flat_model and
                              lease_commence_date):
                with open("./flat_type.json") as file:
                    flat_type_json = json.load(file)
                    for flat_type_name, flat_type_val in flat_type_json.items():
                        if flat_type == flat_type_name:
                            flat_type = flat_type_val
                            break
                with open("./storey_range.json") as file:
                    storey_range_json = json.load(file)
                    for storey_range_name, storey_range_val in storey_range_json.items():
                        if storey_range == storey_range_name:
                            storey_range = storey_range_val
                            break
                with open("./flat_model.json") as file:
                    float_model_json = json.load(file)
                    for float_model_name, float_model_val in float_model_json.items():
                        if flat_model == float_model_name:
                            flat_model = float_model_val
                            break
                # df = CyclicalFeatures().fit_transform(np.array([[flat_year, flat_month, commence_year]])).iloc[:, 3:]
                flat_year_sin = np.sin(flat_year * (2. * np.pi / datetime.datetime.now().year))
                flat_year_cos = np.cos(flat_year * (2. * np.pi / datetime.datetime.now().year))
                flat_month_sin = np.sin(flat_month * (2. * np.pi / 12))
                flat_month_cos = np.cos(flat_month * (2. * np.pi / 12))
                rem_lease = (year_rem_lease * 12) + month_rem_lease
                rem_lease_sin = np.sin(rem_lease * (2. * np.pi / 12))
                rem_lease_cos = np.cos(rem_lease * (2. * np.pi / 12))

                lease_commence_date_sin = np.sin(lease_commence_date * (2. * np.pi / datetime.datetime.now().year - 4))
                lease_commence_date_cos = np.cos(lease_commence_date * (2. * np.pi / datetime.datetime.now().year - 4))
                data = np.array([[flat_type, st.session_state.set_block, storey_range, floor_area_sqm, flat_model,
                                  flat_year_sin, flat_year_cos, flat_month_sin, flat_month_cos, rem_lease_sin,
                                  rem_lease_cos, lease_commence_date_sin, lease_commence_date_cos]])
                prediction = regression_model.predict(data).tolist()[0]
                st.success(f"Based on User Data, Regression model gave to resale price prediction is {prediction}")



