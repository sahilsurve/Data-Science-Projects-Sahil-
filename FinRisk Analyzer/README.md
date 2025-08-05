# 🧠 FinRisk Analyzer – AI-Powered Credit Risk Prediction Tool
<br>

### 📌 Project Description

FinRisk Analyzer is a data driven web-based application that can assist financial institutions to check the likelihood of loan default by borrowers. Using a Random Forest model, the app processes user financial data and delivers real-time, explainable predictions while automating what was traditionally a slow and biased manual process. It features interactive sliders, probability-based outputs, and a feature importance chart to ensure transparency and user education. By effectively and accurately identifying high risk defaults, banks can reduce their loan defaults and lower their financial losses thus reducing the risk of penalties and enhancing regulatory relationships. 

### 🛠 Tech Stack

Frontend: Streamlit <br>
Backend: Python <br>
ML Models: Random Forest <br>
Libraries: Scikit-learn, Pandas, NumPy, Matplotlib

### 💻 How to Run Locally

1. Clone this repo
2. Navigate to the folder directory
3. Install dependencies
4. Run the app using command - streamlit run code.py (I used Anaconda Prompt)

### 📸 Project Demo

Viewing the top feature importance chart
<img width="1903" height="1022" alt="image" src="https://github.com/user-attachments/assets/4494a296-b17c-4e76-8693-1dada4b18a16" />

Adjust your financial information using interactive sliders with description
<img width="1898" height="1091" alt="image" src="https://github.com/user-attachments/assets/aac5ef73-2bde-4d06-a86d-3dc9e9ad5193" />

Viewing the model perforamce metrics and real time risk prediction 
<img width="1918" height="1088" alt="image" src="https://github.com/user-attachments/assets/68796a07-21ff-4af0-912c-fa829c5f5adb" />

### 💡 Most Difficult Challenge

The toughest part was aligning the model choice with the dataset. Although we initially considered using LSTM, the data was static and better suited for Random Forest. Additionally, the dataset required thorough cleaning and preprocessing. Despite our efforts, the model’s accuracy capped at around 74%, which we aimed to improve further.
