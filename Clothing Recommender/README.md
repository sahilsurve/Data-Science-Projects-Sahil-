# 🧠 Clothing Review Search & Classification Web App
<br>

### 📌 Project Description

A Flask-based web application that allows users to:
- 🔍 Search for similar clothing items using **Jaccard similarity**
- 🧠 Classify clothing reviews using **FastText embeddings** + **Logistic Regression**
- 📝 Submit new reviews dynamically 

Built with a combination of **Natural Language Processing (NLP)** techniques and interactive frontend logic, this project demonstrates full-stack data science and intelligent text handling.

## 🚀 Features

- **Search Engine for Clothing Items**  
  Uses stemmed keywords and Jaccard similarity to retrieve similar items based on category or title of clothes.

- **Review Classification**  
  Predicts if a review is positive or negative using pre-trained FastText vectors and a Logistic Regression model.

- **Review Submission Interface**  
  Allows users to submit a new review, which is added to the searchable dataset.

- **Error Handling**  
  Custom 404 and 500 error pages.
  
### 🛠 Tech Stack

- **Backend:** Python, Flask  
- **NLP Libraries:** NLTK, Gensim (FastText), scikit-learn  
- **Similarity:** Jaccard Similarity  
- **Vectorization:** TF-IDF weighted FastText embeddings  
- **ML Model:** Logistic Regression  
- **Frontend:** HTML, Jinja templates, Bootstrap

### 💻 How to Run Locally

1. Clone this repo
2. Navigate to the folder directory in Anaconda prompt
3. Kindly use python version 3.9.5 which can be done by running below command
conda create --name flask_assignment3 python=3.9.5
4. install the required libraries by running below commands
pip install flask
pip install pandas
pip install genism
pip install pickle
pip install nltk
5. Run the app using command - flask run
6. Visit http://127.0.0.1:5000 in your browser

### 📸 Project Demo

Watch the working demo of the app in the video 'Demo_video.mp4'

### 💡 Most Difficult Challenge

A biggest challenge was the extensive data cleaning and preprocessing needed before any modeling could happen, including tokenization, stopword removal, filtering rare and frequent words, and building a clean vocabulary for vectorization. Integrating this with a web interface while maintaining performance and accuracy made it even more complex. If I had more time, I would improve the app's speed by optimizing model loading, and add better error handling and UI enhancements to make it more production-ready.
