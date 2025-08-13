Titanic Survival Prediction 🚢
This project predicts whether a passenger survived the Titanic disaster using Machine Learning. It includes data preprocessing, model training, evaluation, and a Streamlit web application for interactive predictions.

Project Structure:
titanic-app/
├── app.py # Streamlit web application
├── model.pkl # Saved trained ML model
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── data/
│ └── train.csv # Titanic dataset
└── notebooks/
└── model_training.ipynb # Jupyter Notebook for model training

Features:

Data Cleaning: Handle missing values, encode categorical variables, engineer features.

Model Training: Train RandomForestClassifier for prediction.

Model Evaluation: Accuracy score, confusion matrix, and visualizations.

Deployment: Interactive Streamlit app for real-time prediction.

Setup Instructions:

Clone the repository:
git clone https://github.com/Saheela1023/ML_Project.git
cd titanic-app

Create and activate a virtual environment:
Windows:
python -m venv venv
venv\Scripts\activate

Mac/Linux:
python3 -m venv venv
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Dataset:
Download the Titanic dataset from Kaggle: https://www.kaggle.com/c/titanic/data
Place train.csv inside the data/ folder.

Model Training (Jupyter Notebook):

Run: jupyter notebook

Open notebooks/model_training.ipynb

Execute all cells to load and preprocess the data, train the model, and save it as model.pkl.

Running the Streamlit App:

Run: streamlit run app.py

Open the provided local URL in your browser (default: http://localhost:8501)

Input passenger details (Age, Sex, Class, etc.)

Click Predict to see whether the passenger Survived or Did Not Survive.

Example requirements.txt contents:
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
jupyter

License: MIT License

Contact / Acknowledgements:

Email: saheela79@gmail.com

GitHub Repository: https://github.com/MLStreamlit/ML_Project

Dataset: Kaggle Titanic Competition

Libraries: Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit