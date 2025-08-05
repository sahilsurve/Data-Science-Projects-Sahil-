# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import recall_score, f1_score, accuracy_score

# Load the dataset
data = pd.read_csv('heloc_dataset_v1.csv')

# Remove special values
data = data[data['ExternalRiskEstimate'] != -9]
data = data[data['NetFractionRevolvingBurden'] != -8]
data = data[data['MSinceOldestTradeOpen'] != -8]
data = data[data['NumBank2NatlTradesWHighUtilization'] != -8]

# Data split
X = data.drop('RiskPerformance', axis=1)
y = data['RiskPerformance'].map({'Good': 0, 'Bad': 1})

# Feature selection
num_features = 15
model_rfi = RandomForestClassifier(n_estimators=100, random_state=999)
model_rfi.fit(X, y)
fs_indices_rfi = np.argsort(model_rfi.feature_importances_)[::-1][0:num_features]
best_features_rfi = X.columns[fs_indices_rfi].values
X_select = X[best_features_rfi]

# Scaling the selected features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_select)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_select.columns)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df.values, y.values, test_size=0.3, random_state=999)

# Fitting the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=999)
rf_model.fit(X_train, y_train)

# Credit Risk Prediction App Header
st.write("""
# FinRisk Analyzer 🎯 
### Your Instant Credit Risk Predictor 🚀 - Empower Your Financial Decisions! 
This app helps to predict whether someone's credit risk is **Good** or **Bad** using important financial details. 
Simply adjust the values below to make your predictions!
""")
st.write('---')

# Feature importance visualization
feature_importances_rfi = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': X_select.columns,
    'Importance': feature_importances_rfi
}).sort_values(by='Importance', ascending=False)


# Mapping of feature names to user-friendly labels
feature_name_mapping = {
    'ExternalRiskEstimate': 'External Risk Estimate',
    'NetFractionRevolvingBurden': 'Revolving Credit Usage (%)',
    'AverageMInFile': 'Average Months in File',
    'MSinceOldestTradeOpen': 'Months Since Oldest Account Opened',
    'PercentTradesWBalance': 'Percentage of Trades with Balance',
    'PercentInstallTrades': 'Percentage of Installment Trades',
    'NumSatisfactoryTrades': 'Number of Satisfactory Trades',
    'NumTotalTrades': 'Total Number of Trades',
    'MSinceMostRecentInqexcl7days': 'Months Since Most Recent Inquiry',
    'PercentTradesNeverDelq': 'Percentage of Trades Never Delinquent',
    'NetFractionInstallBurden': 'Installment Loan Balance (%)',
    'MSinceMostRecentDelq': 'Months Since Most Recent Delinquency',
    'NumRevolvingTradesWBalance': 'Number of Revolving Trades with Balance',
    'NumBank2NatlTradesWHighUtilization': 'Number of High Utilization Bank/Natl Trades',
    'MSinceMostRecentTradeOpen': 'Months Since Most Recent Account Opened'
}

# Update the feature names in the feature_importances DataFrame using the mapping
feature_imp_df['Feature'] = feature_imp_df['Feature'].map(feature_name_mapping)

# Plot the feature importance with user-friendly feature names
fig, ax = plt.subplots()
ax.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='lightblue')
ax.set_title('Top Features Importance', fontsize=14, color='darkblue')
ax.set_xlabel('Importance', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
st.pyplot(fig)


st.write('---')
st.write("### Input your values using the sliders below and click **Predict**:")
st.write('---')

# Sidebar with collapsible descriptions using st.expander
st.sidebar.header('Input Financial Information 📊')

with st.sidebar.form(key='input_form'):
    
    # External Risk Estimate slider with description in expander
    ERE = st.slider('External Risk Estimate (0-100)', 0, 100, int(X_scaled_df.ExternalRiskEstimate.mean()))
    with st.expander("ℹ️ What is External Risk Estimate?"):
        st.write("This estimates how risky you are based on external factors such as your credit history.")
    
    # Net Fraction Revolving Burden slider with description in expander
    NFRB = st.slider('Net Fraction Revolving Burden (0-300)', 0, 300, int(X_scaled_df.NetFractionRevolvingBurden.mean()))
    with st.expander("ℹ️ What is Net Fraction Revolving Burden?"):
        st.write("This is the percentage of your revolving credit balance compared to your credit limit.")
    
    # Average Months in File slider with description in expander
    AMIF = st.slider('Average Months in File (0-400)', 0, 400, int(X_scaled_df.AverageMInFile.mean()))
    with st.expander("ℹ️ What is Average Months in File?"):
        st.write("This represents how long your credit files have been active, in months.")
    
    # Months Since Oldest Trade Open slider with description in expander
    MSOTO = st.slider('Months Since Oldest Trade Open (0-1000)', 0, 1000, int(X_scaled_df.MSinceOldestTradeOpen.mean()))
    with st.expander("ℹ️ What is Months Since Oldest Trade Open?"):
        st.write("This is the time (in months) since you opened your oldest trade account.")
    
    # Percent Trades with Balance slider with description in expander
    PTWB = st.slider('Percent Trades with Balance (0-100)', 0, 100, int(X_scaled_df.PercentTradesWBalance.mean()))
    with st.expander("ℹ️ What is Percent Trades with Balance?"):
        st.write("The percentage of your trades (accounts) that currently have a balance.")
    
    # Percent Installment Trades slider with description in expander
    PIT = st.slider('Percent Installment Trades (0-100)', 0, 100, int(X_scaled_df.PercentInstallTrades.mean()))
    with st.expander("ℹ️ What is Percent Installment Trades?"):
        st.write("The percentage of your trades that are installment-based, such as loans.")
    
    # Number of Satisfactory Trades slider with description in expander
    NST = st.slider('Number of Satisfactory Trades (0-100)', 0, 100, int(X_scaled_df.NumSatisfactoryTrades.mean()))
    with st.expander("ℹ️ What is Number of Satisfactory Trades?"):
        st.write("The number of your trades (accounts) that have a satisfactory payment history.")
    
    # Number of Total Trades slider with description in expander
    NTT = st.slider('Number of Total Trades (0-150)', 0, 150, int(X_scaled_df.NumTotalTrades.mean()))
    with st.expander("ℹ️ What is Number of Total Trades?"):
        st.write("The total number of trades (accounts) you have on your credit report.")
    
    # Months Since Most Recent Inquiry slider with description in expander
    MSMR7 = st.slider('Months Since Most Recent Inquiry (0-30)', 0, 30, int(X_scaled_df.MSinceMostRecentInqexcl7days.mean()))
    with st.expander("ℹ️ What is Months Since Most Recent Inquiry?"):
        st.write("The time since your most recent credit inquiry, excluding the last 7 days.")
    
    # Percent Trades Never Delinquent slider with description in expander
    PTND = st.slider('Percent Trades Never Delinquent (0-100)', 0, 100, int(X_scaled_df.PercentTradesNeverDelq.mean()))
    with st.expander("ℹ️ What is Percent Trades Never Delinquent?"):
        st.write("The percentage of your trades that have never been delinquent.")
    
    # Net Fraction Install Burden slider with description in expander
    NFIB = st.slider('Net Fraction Install Burden (0-500)', 0, 500, int(X_scaled_df.NetFractionInstallBurden.mean()))
    with st.expander("ℹ️ What is Net Fraction Install Burden?"):
        st.write("The percentage of your installment loan balance.")
    
    # Months Since Most Recent Delinquency slider with description in expander
    MSMRD = st.slider('Months Since Most Recent Delinquency (0-100)', 0, 100, int(X_scaled_df.MSinceMostRecentDelq.mean()))
    with st.expander("ℹ️ What is Months Since Most Recent Delinquency?"):
        st.write("The time since you were last delinquent.")
    
    # Months Since Most Recent Trade Open (missing one)
    MSMRTO = st.slider('Months Since Most Recent Trade Open (0-300)', 0, 300, int(X_scaled_df.MSinceMostRecentTradeOpen.mean()))
    with st.expander("ℹ️ What is Months Since Most Recent Trade Open?"):
        st.write("The time since your most recent trade was opened.")
    
    # Number Revolving Trades with Balance slider with description in expander
    NRTB = st.slider('Number Revolving Trades with Balance (0-50)', 0, 50, int(X_scaled_df.NumRevolvingTradesWBalance.mean()))
    with st.expander("ℹ️ What is Number Revolving Trades with Balance?"):
        st.write("The number of revolving trades (like credit cards) with a balance.")
    
    # Number Bank/Natl Trades with High Utilization Ratio slider with description in expander
    NTHUR = st.slider('Number Bank/Natl Trades with High Utilization Ratio (0-20)', 0, 20, int(X_scaled_df.NumBank2NatlTradesWHighUtilization.mean()))
    with st.expander("ℹ️ What is Number Bank/Natl Trades with High Utilization Ratio?"):
        st.write("The number of bank or national trades with high utilization.")

    # Submit button inside the form
    submit_button = st.form_submit_button(label='Predict Risk Performance')

# Only run the prediction when the form is submitted
if submit_button:
    def user_input_features(): 
        data = {
            'External Risk Estimate': ERE,
            'Net Fraction Revolving Burden': NFRB,
            'Average Months in File': AMIF,
            'Months Since Oldest Trade Open': MSOTO,
            'Percent Trades with Balance': PTWB,
            'Percent Installment Trades': PIT,
            'Number of Satisfactory Trades': NST,
            'Number of Total Trades': NTT,
            'Months Since Most Recent Inquiry': MSMR7,
            'Percent Trades Never Delinquent': PTND,
            'Net Fraction Install Burden': NFIB,
            'Months Since Most Recent Delinquency': MSMRD,
            'Months Since Most Recent Trade Open': MSMRTO,  # Newly added feature
            'Number Revolving Trades with Balance': NRTB,
            'Number Bank/Natl Trades with High Utilization Ratio': NTHUR
        }
        return pd.DataFrame(data, index=[0])

    df = user_input_features()
    
    # Ensure you have all 15 features
    assert df.shape[1] == 15, f"Expected 15 features but got {df.shape[1]}"
    
    # Display selected input parameters
    df_transposed = df.T  # Transpose the DataFrame to switch rows and columns
    df_transposed.columns = ['Values']  # Rename the column for better clarity

    # Display the transposed DataFrame as a vertical table
    st.header('Your Selected Financial Info')
    st.table(df_transposed)

    st.write('---')

    prediction = rf_model.predict(df)
    probability = rf_model.predict_proba(df)

    # Display prediction with color-coded result
    st.header('Risk Prediction Result 🔮')
    if prediction == 0:
        st.error("The risk is predicted to be **Bad** 🚩")
    else:
        st.success("The risk is predicted to be **Good** ✅")
    
    # Display probabilities
    probability_df = pd.DataFrame(probability, columns=['Probability for Bad (0)', 'Probability for Good (1)'])
    st.write(probability_df)
    st.write('---')

# Key Model Metrics Calculation
y_pred = rf_model.predict(X_test) 
recall = recall_score(y_test, y_pred)* 100
f1 = f1_score(y_test, y_pred)* 100
accuracy = accuracy_score(y_test, y_pred)* 100

# Display key metrics
metrics_data = {
    'Metric': ['Recall', 'F1 Score', 'Accuracy'],
    'Value in %': [recall, f1, accuracy]
}
metrics_df = pd.DataFrame(metrics_data)

# Display key metrics in table
st.write("### Model Performance Metrics 📊")
st.table(metrics_df)

####### ROC Curve Visualization
from sklearn.metrics import roc_curve, auc

# Predict probabilities for the test set
y_prob = rf_model.predict_proba(X_test)[:, 1]

