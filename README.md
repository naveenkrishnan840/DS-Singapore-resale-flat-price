# DS-Singapore-Flat-Resale-price

<div align="center">
  <!-- Backend -->
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-3C8D4F?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/MLflow-5C6B8C?style=for-the-badge&logo=mlflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/DagsHub-00A9A6?style=for-the-badge&logo=dagshub&logoColor=white"/>
	
  <h3>Your Singapore Flat Resale price 🚀</h3>

  <p align="center">
    <b> XgBoost | Streamlit | Mlflow | Dagshub  </b>
  </p>
</div>

# Overview
The DS-Singapore-Flat-Resale-Price project is designed to predict the resale price of flats in Singapore using a regression model. The project employs a combination of machine learning, data visualization, and model management tools to predict flat prices based on various features such as flat size, location, age, and amenities. The main goal is to predict flat resale prices accurately and create an interactive environment where users can input different flat characteristics and get predictions in real-time.

# Motivation
The DS-Singapore-Flat-Resale-Price project is motivated by the growing need to predict property prices accurately in a dynamic real estate market, specifically the resale market of public flats in Singapore. several key motivations behind the project like Real Estate Market Uncertainty, Demand for Data-Driven Insights in Real Estate, 
Singapore's Unique Housing Market.

## key Features & Explanation
  1. Regression Model (XGBoost):
       - The core of the project is based on XGBoost, a powerful and widely used machine learning algorithm. XGBoost is a type of gradient boosting method that works well for regression tasks, especially when dealing with complex and large datasets.
       - The model will be trained using various features like:
          - flat_type
          - block
          - storey_range
          - floor_area_sqm
          - flat_model
          - flat_year
          - flat_month
          - remaining_lease
          - lease_commence_date
       - The model will predict the resale price of the flat based on these inputs.
   2. Data Exploration & Visualization:
      	- Plotly and Seaborn will be used to create rich data visualizations that help understand relationships between features and the target variable (resale price).
### **Project Overview: DS-Singapore-Flat-Resale-Price**

The **DS-Singapore-Flat-Resale-Price** project aims to build a **regression model** that predicts the **resale price of flats** in **Singapore** based on a set of key input features. The goal is to leverage **machine learning** techniques to provide accurate price predictions for flats in Singapore’s **HDB resale market**. The project uses a variety of tools and frameworks, including **XGBoost**, **Plotly**, **Streamlit**, **MLflow**, **DagsHub**, and **Seaborn**.

### **Key Components of the Project**

1. **Objective**:
   The project aims to create a regression model that predicts the **resale price** of flats based on various features like flat type, location, size, age, and more. The final goal is to provide a data-driven, predictive tool that can assist potential buyers, sellers, and investors in the Singapore flat resale market.

---

### **Input Features**:
The model takes the following **input features** to predict the resale price:

1. **flat_type**: 
   - The type of flat, such as **1-room**, **2-room**, **3-room**, **4-room**, **5-room**, etc. This feature helps in identifying the size and configuration of the flat.

2. **block**: 
   - The **block number** or identifier where the flat is located. This can provide information on the building's location within a neighborhood.

3. **storey_range**:
   - The **floor range** of the flat, such as **high floor** or **low floor**. This is crucial as higher floor flats tend to have a different market value compared to lower floor flats.

4. **floor_area_sqm**:
   - The **floor area** of the flat in **square meters**. A key determinant of the flat’s price, as larger flats generally cost more.

5. **flat_model**: 
   - The **design/model** of the flat, which could indicate the age or specific layout of the flat (e.g., **standard**, **improved**, **premium**). Different flat models might have different price ranges.

6. **resale_price**:
   - The **target variable** for this regression task. It represents the price at which the flat was sold in the resale market.

7. **flat_year**:
   - The **year of construction** of the flat. Older flats may have a different value compared to newly built flats due to depreciation, maintenance, and other factors.

8. **flat_month**:
   - The **month** in which the flat was sold. This can help identify trends or seasonality in the resale market.

9. **remaining_lease**:
   - The **remaining lease** on the flat. Flats with longer remaining leases typically have higher resale prices as they provide more years of ownership.

10. **lease_commence_date**:
    - The date when the **lease started**, which helps calculate the **remaining lease** and also informs buyers and investors about the flat's long-term value.

---

### **Output Feature**:
The **output feature** of the model is:

- **Resale Price**:
  - The predicted **resale price** of the flat. This is the target variable the model is trying to predict, given the input features.

---

### **Key Technologies and Tools Used**:

1. **XGBoost**:
   - **XGBoost** is a powerful gradient boosting algorithm used for **regression** tasks. It is the core of the model and will be used to train the predictive model that estimates the resale price based on the input features.
   - The benefits of **XGBoost** include:
     - Ability to handle **large datasets** efficiently.
     - Robust against **overfitting**.
     - Works well with **both categorical and numerical** data, making it ideal for this project.
     - Hyperparameter tuning to improve model performance and accuracy.

2. **Plotly**:
   - **Plotly** is a visualization tool that helps to create **interactive visualizations**. It will be used to create plots that show the relationships between the different features and the resale price, including:
     - Scatter plots to visualize the impact of features like **floor area** or **storey range** on **resale price**.
     - Heatmaps to visualize the correlations between features.
     - Interactive plots where users can explore the dataset and predictions.

3. **Streamlit**:
   - **Streamlit** will be used to build an **interactive web application** for real-time prediction. Users will be able to input various parameters such as **flat type**, **floor area**, and **remaining lease**, and the model will provide an estimated resale price.
   - The **Streamlit app** will display:
     - Real-time predictions of resale prices.
     - Visualizations and graphs for better understanding of the model.
     - Performance metrics to show how accurate the model is (e.g., RMSE, R²).

4. **MLflow**:
   - **MLflow** will be used for **model tracking** and **experiment management**. It will help to:
     - Track the performance of different models and hyperparameters.
     - Store and version models, allowing easy rollback or future improvements.
     - Record metrics like **RMSE**, **R²**, and other evaluation measures during model training.

5. **DagsHub**:
   - **DagsHub** is a platform for versioning and managing machine learning projects. It will be used to:
     - Store the dataset.
     - Version control the data and code.
     - Collaborate with other team members, allowing for easier sharing and collaboration.

6. **Seaborn**:
   - **Seaborn** is a statistical data visualization library that will be used for:
     - Creating **distribution plots** and **histograms** to understand the spread of the resale prices.
     - Plotting **correlation heatmaps** to identify key relationships between features and resale prices.
     - Generating **pair plots** to visualize multiple variables at once.

---

### **Project Workflow**:

1. **Data Preprocessing**:
   - Clean and preprocess the data, including handling **missing values**, encoding **categorical variables** (e.g., flat type, flat model), and scaling the **numerical features** like floor area and remaining lease.
   
2. **Exploratory Data Analysis (EDA)**:
   - Use **Seaborn** and **Plotly** to explore the relationships between features and resale prices.
   - Identify important correlations and trends, such as whether **floor area** or **flat type** is most strongly correlated with price.
   
3. **Model Development**:
   - Train an **XGBoost regression model** using the input features to predict the **resale price**.
   - Tune hyperparameters to optimize model performance.

4. **Model Evaluation**:
   - Evaluate the model using performance metrics such as **RMSE** (Root Mean Squared Error) and **R²** (coefficient of determination).
   - Use **MLflow** to track different experiment versions and monitor model performance.

5. **Deployment**:
   - Deploy the model as an interactive web app using **Streamlit**, where users can input features and get real-time predictions.
   - Visualize key insights and predictions through **interactive charts**.

---

## Project Structure
```
Youtube-Video-Summarization/
├── backend/
|  ├── src/
|  |   ├── audio_path/
|  │   ├── documents/                           
|  │   │   ├── image.png                     
|  │   │   ├── audio_text.txt             
|  |   |── lancedb/
|  │   │   ├── image_collections.lance/
|  │   │   ├── text_collections.lance/
|  |   |── tokenizer_path/         
|  |   ├── video_path/
|  |   |── whisper_model_path/
|  │   ├── request_validate.py          
|  |   ├── video_extract.py
|  │────── .env # If you want
|  │────── pyproject.toml # create virtual env using poetry
|  │────── main.py # Main entry point
|  ├── pyproject.toml
|  ├── ...
```

## Setup Instructions

### Backend Setup

1. Clone the repository
   ```bash
   git clone https://github.com/naveenkrishnan840/Youtube-Video-Summarization.git
   cd Youtube-Video-Summarization
   cd backend
   ```

2. Install Poetry (if not already installed)

   Mac/Linux:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   Windows:
   ```bash
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   ```

3. Set Python version for Poetry
   ```bash
   poetry env use python3.12
   ```

4. Activate the Poetry shell:
   For Unix/Linux/MacOS:
   ```bash
   poetry shell
   # or manually
   source $(poetry env info --path)/bin/activate
   ```
   For Windows:
   ```bash
   poetry shell
   # or manually
   & (poetry env info --path)\Scripts\activate
   ```

5. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

6. Set up environment variables in `.env`:
   ```bash
    DOCUMENTS_PATH="your docs path"
    AUDIO_PATH="your audio path"
    IMAGE_FORMAT=frame%04d.png
    AUDIO_FORMAT=output_audio.mp3
    VIDEO_PATH="your video path"
    AUDIO_TEXT_FORMAT=audio_text.txt
    OUTPUT_AUDIO_PATH=output_text.txt
    GOOGLE_API_KEY="you api here"
    LANCEDB_PATH="your DB path"
   ```

7. Run the backend:

   Make sure you are in the backend folder

    ```bash
    uvicorn app.main:app --reload --port 8000 
    ```

   For Windows User:

    ```bash
    uvicorn app.main:app --port 8000
    ```

8. Access the API at `http://localhost:8000`

### Frontend Setup

1. Open a new terminal and make sure you are in the WebRover folder:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the frontend:
   ```bash
   npm run dev
   ```

4. Access the frontend at `http://localhost:3000`

For mac users: 

Try running http://localhost:3000 on Safari browser. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [@naveenkrishnan840](https://github.com/naveenkrishnan840)
