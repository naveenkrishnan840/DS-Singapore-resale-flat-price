import os
import pandas as pd
import streamlit
import streamlit as st
import plotly.express as px
import json
import numpy as np
import scipy.stats as stat
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from feature_engine.creation import CyclicalFeatures

st.set_page_config(layout="wide", page_title="Copper Set Modelling Machine Learning Task")

st.markdown("<h2 style='font-family: san-serif; color: red; text-align: center;'>"
            " Singapore Resale Flat Price </h2>", unsafe_allow_html=True)
task = st.selectbox(label="Select ML Task", options=["Select", "Regression"])


if task != "Select":
    before_modeling_training = True
    modeling_training = False
    st.divider()


    @st.cache_data
    def load_csv():
        return pd.read_csv("./Resale-flat-prices.csv", low_memory=False)
    regression_data = load_csv()

    @st.cache_resource
    def render_chart(data_frame):
        st.plotly_chart(px.bar(data_frame, x=data_frame.index, y="proportion"), theme=None)

    @st.cache_data
    def display_plot(series, label):
        st.plotly_chart(ff.create_distplot([series], group_labels=label), theme=None)

    @st.cache_resource
    def render_box_chart(data_frame, column):
        st.plotly_chart(px.box(data_frame, x=column), theme=None)
    st.markdown("<style>p:hover{background-color: grey;}</style>", unsafe_allow_html=True)
    if task == "Regression":
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Data Ingestion </p>",
                        unsafe_allow_html=True)
            st.dataframe(regression_data.head(10), use_container_width=True)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Show Missing & Nan Count </p>",
                        unsafe_allow_html=True)
            st.dataframe(regression_data.isnull().sum(), use_container_width=True)
            st.markdown("<h5 style='color: red;'> There is no missing values. We will proceed further steps </h5>",
                        unsafe_allow_html=True)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Segregate the feature </p>",
                        unsafe_allow_html=True)
            cols = st.columns(2)
            with cols[0]:
                st.markdown("<h5 style='color: red;'> Numerical Feature </h5>", unsafe_allow_html=True)
                cate_df = regression_data.select_dtypes(include=[np.number])
                st.dataframe(cate_df)
            with cols[1]:
                st.markdown("<h5 style='color: red;'> Categorical Feature </h5>", unsafe_allow_html=True)
                num_df = regression_data.select_dtypes(exclude=[np.number])
                st.dataframe(num_df)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > "
                        "Visualize the categorical features</p>",
                        unsafe_allow_html=True)
            flat_type = pd.DataFrame(regression_data["flat_type"].value_counts(normalize=True))
            st.markdown("<h5 style='color: red;'>Flat Type feature</h5>", unsafe_allow_html=True)
            render_chart(flat_type)
            flat_model = pd.DataFrame(regression_data["flat_model"].value_counts(normalize=True))
            st.markdown("<h5 style='color: red;'>Flat Model feature</h5>", unsafe_allow_html=True)
            render_chart(flat_model)
            LabelEncoder().fit_transform(regression_data["block"])
            block_df = pd.DataFrame(regression_data["block"].value_counts(normalize=True))
            st.markdown("<h5 style='color: red;'>Block feature</h5>", unsafe_allow_html=True)
            render_chart(block_df)
            storey_range_df = regression_data["storey_range"].value_counts(normalize=True)
            st.markdown("<h5 style='color: red;'>Storey Range feature</h5>", unsafe_allow_html=True)
            render_chart(storey_range_df)
            # st.plotly_chart(px.bar(storey_range_df, x=storey_range_df.index, y="proportion"), theme=None)

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' >"
                        "Handle Categorical feature</p>",
                        unsafe_allow_html=True)
            flat_type_dict = regression_data["flat_type"].value_counts(normalize=True).to_dict()
            regression_data["flat_type"] = regression_data["flat_type"].map(flat_type_dict)
            if os.path.exists("./flat_type.json"):
                os.remove("./flat_type.json")
            with open("./flat_type.json", "w") as file:
                file.write(json.dumps(flat_type_dict))

            flat_model_dict = regression_data["flat_model"].value_counts(normalize=True).to_dict()
            regression_data["flat_model"] = regression_data["flat_model"].map(flat_model_dict)
            if os.path.exists("./flat_model.json"):
                os.remove("./flat_model.json")
            with open("./flat_model.json", "w") as file:
                file.write(json.dumps(flat_model_dict))

            block_dict = regression_data["block"].value_counts(normalize=True).to_dict()
            regression_data["block"] = regression_data["block"].map(block_dict)
            if os.path.exists("./block.json"):
                os.remove("./block.json")
            with open("./block.json", "w") as file:
                file.write(json.dumps(block_dict))
            storey_range_dict = regression_data["storey_range"].value_counts(normalize=True).to_dict()
            regression_data["storey_range"] = regression_data["storey_range"].map(storey_range_dict)
            if os.path.exists("./storey_range.json"):
                os.remove("./storey_range.json")
            with open("./storey_range.json", "w") as file:
                file.write(json.dumps(storey_range_dict))
            st.dataframe(regression_data.head(10), use_container_width=True)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Handle Date, Month, Year</p>",
                        unsafe_allow_html=True)
            regression_data["flat_year"] = regression_data["month"].str.split("-").apply(
                lambda x: x[0]).astype(int)
            regression_data["flat_month"] = regression_data["month"].str.split("-").apply(
                lambda x: x[0]).astype(int)
            regression_data["remaining_lease"] = regression_data["remaining_lease"].str.split(" ").apply(
                lambda x: int(x[0]) * 12 + (int(x[2]) if len(x) == 4 else 0))
            regression_data.drop(labels=["town", "street_name", "month"], axis=1, inplace=True)
            st.dataframe(regression_data.head(10), use_container_width=True)

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Identify Skewness </p>",
                        unsafe_allow_html=True)

            st.markdown("<h5 style='color: red;'> Floor Area Sqm </h5>", unsafe_allow_html=True)
            CLT_floor_area_sqm_data = pd.Series([regression_data.loc[:, "floor_area_sqm"].sample(n=100,
                                                                                                 replace=True).mean()
                                        for i in range(0, 1000)])
            st.markdown(f"<h5 style='color: red;'> skewness Of Floor Area Sqm is"
                        f" {str(CLT_floor_area_sqm_data.skew())} </h5>", unsafe_allow_html=True)
            display_plot(CLT_floor_area_sqm_data, ["Floor Area Sqm"])
            # st.plotly_chart(ff.create_distplot([regression_data['floor_area_sqm']],
            #                                    group_labels=["Floor Area Sqm"]), theme=None)
            st.markdown("<h5 style='color: red;'> Positive Skewness is asymmetrical distribution "
                        "(mean > median > mode)</h5>", unsafe_allow_html=True)
            st.text(f"check the condition for positive or right skewness (mean > median > mode) is "
                    f"{CLT_floor_area_sqm_data.mean()} > {CLT_floor_area_sqm_data.median()} > "
                    f"{stat.mode(CLT_floor_area_sqm_data).mode}")
            st.markdown("<h5 style='color: red;'>Lease Commence Date</h5>", unsafe_allow_html=True)
            CLT_lease_commence_date_data = pd.Series([regression_data.loc[:, "lease_commence_date"].sample(n=100,
                                                                                                 replace=True).mean()
                                                 for i in range(0, 1000)])
            st.markdown(f"<h5 style='color: red;'>skewness Of Lease Commence Date is "
                        f"{str(CLT_lease_commence_date_data.skew())}</h5>", unsafe_allow_html=True)
            display_plot(CLT_lease_commence_date_data, ["Lease Commence Date"])
            # st.plotly_chart(ff.create_distplot([regression_data['lease_commence_date']],
            #                                    group_labels=["lease_commence_date"]), theme=None)
            st.markdown("<h5 style='color: red;'>Positive Skewness is asymmetrical distribution "
                        "(mean > median > mode) </h5>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='color: red;'>check the condition for positive or right skewness "
                        f"(mean > median > mode) is "
                        f"{CLT_lease_commence_date_data.mean()} > {CLT_lease_commence_date_data.median()} > "
                        f"{stat.mode(CLT_lease_commence_date_data).mode}</h5", unsafe_allow_html=True)

            st.markdown("<h5 style='color: red;'> Remaining Lease </h5>", unsafe_allow_html=True)
            CLT_remaining_lease_data = pd.Series([regression_data.loc[:, "remaining_lease"].sample(n=100,
                                                                                                           replace=True).mean()
                                                      for i in range(0, 1000)])
            st.markdown(f"<h5 style='color: red;' >skewness Of Remaining Lease is "
                        f"{str(CLT_remaining_lease_data.skew())} </h5>", unsafe_allow_html=True)
            display_plot(CLT_remaining_lease_data, ["Remaining Lease"])
            # st.plotly_chart(ff.create_distplot([regression_data['remaining_lease']],
            #                                    group_labels=["Remaining Lease"]), theme=None)
            st.markdown("Positive Skewness is asymmetrical distribution (mean > median > mode)")
            st.markdown(f"<h5 style='color: red;'>check the condition for positive or right skewness "
                        f"(mean > median > mode) is "
                        f"{CLT_remaining_lease_data.mean()} > "
                        f"{CLT_remaining_lease_data.median()} > "
                        f"{stat.mode(CLT_remaining_lease_data).mode}</h5", unsafe_allow_html=True)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem;' > "
                        "showing outlier & handling outlier </p>", unsafe_allow_html=True)
            st.markdown("<h4> Before Handling Outlier </h4>", unsafe_allow_html=True)
            st.markdown("<h5> Box Plot </h5>", unsafe_allow_html=True)
            render_box_chart(regression_data, "flat_year")
            # st.plotly_chart(px.box(regression_data, x="flat_year"), theme=None)
            # st.plotly_chart(px.box(regression_data, x="flat_month"), theme=None)
            render_box_chart(regression_data, "flat_month")
            # st.plotly_chart(px.box(regression_data, x="floor_area_sqm"), theme=None)
            render_box_chart(regression_data, "floor_area_sqm")
            # st.plotly_chart(px.box(regression_data, x="lease_commence_date"), theme=None)
            render_box_chart(regression_data, "lease_commence_date")
            # st.plotly_chart(px.box(regression_data, x="remaining_lease"), theme=None)
            render_box_chart(regression_data, "remaining_lease")
            lower_floor_area_sqm_limit = regression_data["floor_area_sqm"].mean() - 3 * (
                regression_data["floor_area_sqm"].std())
            upper_floor_area_sqm_limit = regression_data["floor_area_sqm"].mean() + 3 * (
                regression_data["floor_area_sqm"].std())
#
            # Count of Width Outlier
            count_of_outlier = regression_data[
                (regression_data["floor_area_sqm"] < lower_floor_area_sqm_limit) |
                (regression_data["floor_area_sqm"] > upper_floor_area_sqm_limit)].shape[0]
            st.text(f"Count Of Floor Area Sqm Outlier is {str(count_of_outlier)}")
            # Handling Width Outlier
            regression_data["floor_area_sqm"] = np.where(regression_data["floor_area_sqm"] <
                                                         lower_floor_area_sqm_limit,
                                                         lower_floor_area_sqm_limit,
                                                         np.where(regression_data["floor_area_sqm"] >
                                                         upper_floor_area_sqm_limit,
                                                         upper_floor_area_sqm_limit,
                                                         regression_data["floor_area_sqm"]))
            st.markdown("<h5> After handling Outlier for floor area sqm </h5>", unsafe_allow_html=True)
            render_box_chart(regression_data, "floor_area_sqm")
            # st.plotly_chart(px.box(regression_data, x="floor_area_sqm"), theme=None)
#
#             lower_thickness_limit = regression_data["thickness"].mean() - 3 * regression_data["thickness"].std()
#             upper_thickness_limit = regression_data["thickness"].mean() + 3 * regression_data["thickness"].std()
#             # Count of thickness Outlier
#             count_of_outlier = regression_data[
#                 (regression_data["thickness"] < lower_thickness_limit) |
#                 (regression_data["thickness"] > upper_thickness_limit)].shape[0]
#             st.text(f"Count Of thickness Outlier is {str(count_of_outlier)}")
#             # Handling thickness Outlier
#             regression_data["thickness"] = np.where(regression_data["thickness"] < lower_thickness_limit,
#                                                     lower_thickness_limit,
#                                                     np.where(regression_data["thickness"] > upper_thickness_limit,
#                                                              upper_thickness_limit, regression_data["thickness"]))
#             st.markdown("<h4> After Handling Outlier </h4>", unsafe_allow_html=True)
#             st.markdown("<h5> Violin Plot </h5>", unsafe_allow_html=True)
#             st.plotly_chart(px.violin(regression_data, x="width", box=True), theme=None)
#             st.plotly_chart(px.violin(regression_data, x="thickness", box=True), theme=None)
#             st.markdown("<h5> Box Plot </h5>", unsafe_allow_html=True)
#             st.plotly_chart(px.box(regression_data, x="width"), theme=None)
#             st.plotly_chart(px.box(regression_data, x="thickness"), theme=None)
#             st.markdown("<h5> Correlation Chart </h5>", unsafe_allow_html=True)
#             st.plotly_chart(px.imshow((regression_data.corr()), text_auto=True, height=1000), theme=None)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Applying Cyclical Features"
                        "</p>", unsafe_allow_html=True)
            cyclical_features = CyclicalFeatures().fit_transform(
                regression_data[["flat_year", "flat_month", "remaining_lease", "lease_commence_date"]])
            cyclical_features = cyclical_features[["flat_year_sin", "flat_year_cos", "flat_month_sin",
                                                   "flat_month_cos",
                                                   "remaining_lease_sin", "remaining_lease_cos",
                                                   'lease_commence_date_sin', "lease_commence_date_cos"]]
            regression_data[cyclical_features.columns] = cyclical_features
            regression_data.drop(labels=["lease_commence_date", "remaining_lease", "flat_year", "flat_month"], axis=1,
                                 inplace=True)
            st.dataframe(regression_data.head(10), use_container_width=True)
            st.markdown("<h5 style='color: red;'> Correlation Chart </h5>", unsafe_allow_html=True)
            st.plotly_chart(px.imshow((regression_data.corr()), text_auto=True, height=1000), theme=None)
            X = regression_data.drop(labels=["resale_price"], axis=1)
            y = regression_data["resale_price"]

        with (st.container()):
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; '>"
                        "Scaling & Split The Data</p>", unsafe_allow_html=True)
            # st.text(",".join(X.columns))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train[X_train.columns] = scaler.transform(X_train)
            X_test[X_test.columns] = scaler.transform(X_test)
            X_train.reset_index(drop=True, inplace=True)
            X_test.reset_index(drop=True, inplace=True)
            st.markdown("<h5> Scaling Data </h5>", unsafe_allow_html=True)
            st.dataframe(pd.concat([X, y], axis=1).head(10), use_container_width=True)
            X_train["resale_price"] = y_train
            X_test["resale_price"] = y_test
            if not os.path.exists("./regression"):
                os.makedirs("regression", exist_ok=True)
            if os.path.exists("./regression/training_data.csv"):
                os.remove("./regression/training_data.csv")
            if os.path.exists("./regression/testing_data.csv"):
                os.remove("./regression/testing_data.csv")
            X_train.to_csv("./regression/training_data.csv", index=False)
            X_test.to_csv("./regression/testing_data.csv", index=False)
            data_count_cols = st.columns(2, vertical_alignment="center")
            with data_count_cols[0]:
                st.markdown("<h5 style='color: red;'> Training Data Count is " + str(X_train.shape[0]) + " </h5>",
                            unsafe_allow_html=True)
            with data_count_cols[1]:
                st.markdown("<h5 style='color: red;'> Testing Data Count is " + str(X_test.shape[0]) + " </h5>",
                            unsafe_allow_html=True)
#
# # st.plotly_chart(px.imshow(regression_data.corr(), text_auto=True))
# # # st.markdown("<h5> Showing HeatMap Chart </h5>", unsafe_allow_html=True)
# # #             st.plotly_chart(px.density_heatmap(classification_data.corr(), text_auto=True))
