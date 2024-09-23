import os
import pandas as pd
import streamlit
import streamlit as st
import plotly.express as px
import seaborn as sns
import json
# import matplotlib.pyplot as plt
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

from mlflow.models import infer_signature
import mlflow
st.set_page_config(layout="wide", page_title="Copper Set Modelling Machine Learning Task")
# st.container()

st.markdown("<h2 style='font-family: san-serif; color: red; text-align: center;'>"
            " Singapore Resale Flat Price </h2>", unsafe_allow_html=True)

task = st.selectbox(label="Select ML Task", options=["Select", "Regression"])
# st.markdown("", unsafe_allow_html=True)
if task != "Select":
    before_modeling_training = True
    modeling_training = False
    st.divider()


    @st.cache_data
    def load_csv():
        return pd.read_excel("./Resale-flat-prices.csv")
    regression_data = load_csv()

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
            regression_data.drop_duplicates(inplace=True, keep="first")
            st.dataframe(regression_data.isnull().sum(), use_container_width=True)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Handle Missing Values </p>",
                        unsafe_allow_html=True)
            regression_data["delivery date"] = regression_data["delivery date"].fillna(
                regression_data["delivery date"].mode()[0])
            regression_data["selling_price"] = regression_data["selling_price"].fillna(
                regression_data["selling_price"].median())
            regression_data["thickness"] = regression_data["thickness"].fillna(regression_data["thickness"].median())
            regression_data["application"] = regression_data["application"].fillna(
                regression_data["application"].mode()[0])
            regression_data["delivery_date_year"] = regression_data["delivery date"].astype(int).astype(str).str.slice(
                0, 4).astype(int)
            regression_data["delivery_date_month"] = regression_data["delivery date"].astype(int).astype(str).str.slice(
                4, 6).astype(int)
            regression_data["delivery_date_day"] = regression_data["delivery date"].astype(int).astype(str).str.slice(
                6, 8).astype(int)
            regression_data = regression_data[~(regression_data["delivery_date_month"] == 22)]
            regression_data.drop(labels=["product_ref", "material_ref", "delivery date"], axis=1, inplace=True)
            st.dataframe(regression_data.head(10), use_container_width=True)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' >Visualize the categorical features</p>",
                        unsafe_allow_html=True)
            cate_item = pd.DataFrame(regression_data["item type"].value_counts())
            st.markdown("<h5 style='color: red;'>Item Type feature</h5>", unsafe_allow_html=True)
            st.plotly_chart(px.bar(cate_item, x=cate_item.index, y="count"), theme=None)
            application = pd.DataFrame(regression_data["application"].value_counts())
            st.markdown("<h5 style='color: red;'>Application feature</h5>", unsafe_allow_html=True)
            st.plotly_chart(px.bar(application, x=application.index, y="count"), theme=None)

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' >Handle Categorical feature</p>",
                        unsafe_allow_html=True)
            item_type_class_encoder_dict = {j: i + 1 for i, j in regression_data[
                "item type"].drop_duplicates().reset_index(drop=["index"]).items()}
            regression_data["item type"] = regression_data["item type"].apply(
                lambda x: item_type_class_encoder_dict[x])
            if os.path.exists("./item_type.json"):
                os.remove("./item_type.json")
            with open("./item_type.json", "w") as file:
                file.write(json.dumps(item_type_class_encoder_dict))
            st.dataframe(regression_data.head(10), use_container_width=True)

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Identify Skewness as distplot </p>",
                        unsafe_allow_html=True)
            CLT_width_data = pd.Series([regression_data.loc[:, "width"].sample(n=100, replace=True).mean()
                                        for i in range(0, 500)])
            st.text(f"skewness Of Width is {str(CLT_width_data.skew())}")
            st.plotly_chart(ff.create_distplot([CLT_width_data], group_labels=["Width"]), theme=None)
            st.text("Negative Skewness is asymmetrical distribution (median > mean  > mode)")
            st.text(f"check the condition for negative or left skewness (median > mean > mode) is "
                    f"{CLT_width_data.median()} > {CLT_width_data.mean()} > "
                    f"{stat.mode(CLT_width_data).mode}")

            CLT_thickness_data = pd.Series([regression_data.loc[:, "thickness"].sample(n=100, replace=True).mean()
                                            for i in range(0, 1000)])
            st.text(f"skewness Of Thickness is {str(CLT_thickness_data.skew())}")
            st.plotly_chart(ff.create_distplot([CLT_thickness_data], group_labels=["Thickness"]), theme=None)
            st.text("Positive Skewness is asymmetrical distribution (mean > median > mode)")
            st.markdown(f"<h5 style='color: red;'>check the condition for positive or right skewness (mean > median > mode) is "
                        f"{CLT_thickness_data.mean()} > {CLT_thickness_data.median()} > "
                        f"{stat.mode(CLT_thickness_data).mode}</h5", unsafe_allow_html=True)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem;' > "
                        "showing outlier & handling outlier </p>", unsafe_allow_html=True)
            st.markdown("<h4> Before Handling Outlier </h4>", unsafe_allow_html=True)
            st.markdown("<h5> Violin Plot </h5>", unsafe_allow_html=True)
            st.plotly_chart(px.violin(regression_data, x="width", box=True), theme=None)
            st.plotly_chart(px.violin(regression_data, x="thickness", box=True), theme=None)
            st.markdown("<h5> Box Plot </h5>", unsafe_allow_html=True)
            st.plotly_chart(px.box(regression_data, x="width"), theme=None)
            st.plotly_chart(px.box(regression_data, x="thickness"), theme=None)
            lower_width_limit = regression_data["width"].mean() - 3 * regression_data["width"].std()
            upper_width_limit = regression_data["width"].mean() + 3 * regression_data["width"].std()

            # Count of Width Outlier
            count_of_outlier = regression_data[
                (regression_data["width"] < lower_width_limit) |
                (regression_data["width"] > upper_width_limit)].shape[0]
            st.text(f"Count Of Width Outlier is {str(count_of_outlier)}")
            # Handling Width Outlier
            regression_data["width"] = np.where(regression_data["width"] < lower_width_limit,
                                                lower_width_limit,
                                                np.where(regression_data["width"] > upper_width_limit,
                                                         upper_width_limit, regression_data["width"]))

            lower_thickness_limit = regression_data["thickness"].mean() - 3 * regression_data["thickness"].std()
            upper_thickness_limit = regression_data["thickness"].mean() + 3 * regression_data["thickness"].std()
            # Count of thickness Outlier
            count_of_outlier = regression_data[
                (regression_data["thickness"] < lower_thickness_limit) |
                (regression_data["thickness"] > upper_thickness_limit)].shape[0]
            st.text(f"Count Of thickness Outlier is {str(count_of_outlier)}")
            # Handling thickness Outlier
            regression_data["thickness"] = np.where(regression_data["thickness"] < lower_thickness_limit,
                                                    lower_thickness_limit,
                                                    np.where(regression_data["thickness"] > upper_thickness_limit,
                                                             upper_thickness_limit, regression_data["thickness"]))
            st.markdown("<h4> After Handling Outlier </h4>", unsafe_allow_html=True)
            st.markdown("<h5> Violin Plot </h5>", unsafe_allow_html=True)
            st.plotly_chart(px.violin(regression_data, x="width", box=True), theme=None)
            st.plotly_chart(px.violin(regression_data, x="thickness", box=True), theme=None)
            st.markdown("<h5> Box Plot </h5>", unsafe_allow_html=True)
            st.plotly_chart(px.box(regression_data, x="width"), theme=None)
            st.plotly_chart(px.box(regression_data, x="thickness"), theme=None)
            st.markdown("<h5> Correlation Chart </h5>", unsafe_allow_html=True)
            st.plotly_chart(px.imshow((regression_data.corr()), text_auto=True, height=1000), theme=None)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Applying Power Transformation "
                        "</p>", unsafe_allow_html=True)
            regression_data[["width", "thickness"]] = FunctionTransformer(func=np.log1p).fit_transform(
                regression_data[["width", "thickness"]])
            st.dataframe(regression_data.head(10), use_container_width=True)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; '> "
                        "Feature Selection By Variance Threshold</p>", unsafe_allow_html=True)
            X = regression_data.drop(labels=["selling_price"], axis=1)
            y = regression_data["selling_price"]
            variance_threshold = VarianceThreshold(threshold=0.3)
            variance_threshold.fit_transform(X=X, y=y)
            variance_threshold_df = pd.DataFrame(variance_threshold.feature_names_in_)
            variance_threshold_df.columns = ["Columns"]
            variance_threshold_df["Variance"] = variance_threshold.get_support()
            variance_threshold_df["Variance"] = variance_threshold_df["Variance"].apply(
                lambda x: "High Variance" if x else "Low Variance")
            st.table(variance_threshold_df)
            X = X[X.columns[variance_threshold.get_support()]]
            st.markdown("<h5> After applying Feature Selection </h5>", unsafe_allow_html=True)
            st.dataframe(X.head(10), use_container_width=True)
            # regression_data = pd.concat([X, y], axis=1)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; '>"
                        "Scaling & Split The Data</p>", unsafe_allow_html=True)
            # st.text(",".join(X.columns))
            X[X.columns.to_list()] = StandardScaler().fit_transform(X)
            st.markdown("<h5> Scaling Data </h5>", unsafe_allow_html=True)
            st.dataframe(pd.concat([X, y], axis=1).head(10), use_container_width=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            X_train["selling_price"] = y_train
            X_test["selling_price"] = y_test
            if not os.path.exists("./regression"):
                os.makedirs("regression")
            if os.path.exists("./regression/training_data.csv"):
                os.remove("./regression/training_data.csv")
            if os.path.exists("./regression/testing_data.csv"):
                os.remove("./regression/testing_data.csv")
            X_train.to_csv("./regression/training_data.csv", index=False)
            X_test.to_csv("./regression/testing_data.csv", index=False)
            data_count_cols = st.columns(2, vertical_alignment="center")
            with data_count_cols[0]:
                st.markdown("<h5> Training Data Count is " + str(X_train.shape[0]) + " </h5>", unsafe_allow_html=True)
            with data_count_cols[1]:
                st.markdown("<h5> Testing Data Count is " + str(X_test.shape[0]) + " </h5>", unsafe_allow_html=True)

# st.plotly_chart(px.imshow(regression_data.corr(), text_auto=True))
# # st.markdown("<h5> Showing HeatMap Chart </h5>", unsafe_allow_html=True)
# #             st.plotly_chart(px.density_heatmap(classification_data.corr(), text_auto=True))
