# Import required libraries
from flask import Flask, render_template, request
import pandas as pd
from gensim.models.fasttext import FastText
import pickle
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Import data from CSV and parse it into dataframe
og_data = pd.read_csv('assignment3_II.csv')    
og_data['ID'] = range(1, len(og_data) + 1)

# Create a copy of original dataset
data = og_data.copy()

# Perform stemming and lower case all data in 'Clothes Title' and 'Class Name'
stemmer = PorterStemmer()
ClothesTitle = data['Clothes Title'].apply(lambda text: stemmer.stem(text.lower()))
ClassName = data['Class Name'].apply(lambda text: stemmer.stem(text.lower()))

# Create new columns for processed Clothes Title and Class Name
data['Processed Class Name'] = ClassName
data['Processed Clothes Title'] = ClothesTitle
# Joining them to use both searching
data['Processed Clothes Info'] = data['Processed Class Name'] + ' ' + data['Processed Clothes Title'] 

# Function to calculate jaccard's similarity
def jaccard_similarity(str1, str2):
    tokens1 = set(str1.split())
    tokens2 = set(str2.split())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if len(union) > 0 else 0

# Function to find items similar to the searched item
def similar_clothing_items(item_searched):
    # Lower case the searched item
    item_searched = item_searched.lower()

    # Perform stemming
    item_searched = stemmer.stem(item_searched)

    data_copy = data.copy()
    # Calculate similarity for the combined Class name and Cloth title
    data_copy['Similarity'] = data_copy['Processed Clothes Info'].apply(lambda x: jaccard_similarity(item_searched, x))
    
    # Filter out the items that have similarity > 0
    filtered_data = data_copy[data_copy['Similarity'] > 0]

    # Sort by similarity in descending order
    sorted_items = filtered_data.sort_values(by='Similarity', ascending=False)
    
    # Group by Clothing ID to get unique items
    unique_clothing_items = sorted_items.drop_duplicates(subset='Clothing ID')

    return unique_clothing_items

# Define the route for home page
@app.route('/')
def index():
    # Render the base template
    return render_template('base.html', show_footer = True)

# Define the route for search page, accepting both GET and POST requests
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        # Get the search input from the form
        f_search = request.form['item_searched']
        
        # Find similar items to the searched item
        similar_items = similar_clothing_items(f_search)
        
        # Get the number of matched items
        n_matched_items = len(similar_items)
        
        # Render the search page with matched items and count
        return render_template('search.html', 
                               items=similar_items[['Clothing ID', 'Class Name', 'Clothes Title']].to_dict(orient='records'),
                               n_matched_items=n_matched_items,
                               search_value=f_search)
        
    # Default case when the page is loaded without search
    return render_template('search.html', items=[], n_matched_items=0, search_value="")

# Define the route for item details page wrt clothing ID
@app.route('/clothing/<int:clothing_id>')
def item_detail(clothing_id):
    # Find all reviews for the given Clothing ID
    all_reviews = og_data[og_data['Clothing ID'] == clothing_id].to_dict(orient='records')

    # Render the detailed page with item details
    return render_template('item_details.html', item=all_reviews)

# Define the route for classify page, accepting both GET and POST requests
@app.route('/classify', methods=['GET', 'POST'])
def classify():
        if request.method == 'POST':

            # Read the content from the user's input
            f_title = request.form['title']
            f_content = request.form['description']
            f_rating = request.form['rating']
            f_recommend = request.form.get('recommend', '')
            f_id = request.form['clothing_id']
            f_class = request.form['class_name']

            # Check if it's a classification or submission action
            if 'classify' in request.form:

                global data, og_data

                # Preprocess the inputed review title and description

                pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
                tokenizer = RegexpTokenizer(pattern)

                # Tokenize and lower case the review title and description
                token_ftitle = tokenizer.tokenize(f_title.lower())
                token_fcontent = tokenizer.tokenize(f_content.lower())

                # Remove the single characters 
                single_ftitle = [word for word in token_ftitle if len(word) >= 2]
                single_fcontent = [word for word in token_fcontent if len(word) >= 2]

                # Read stop words from the stopwords_en.txt file
                stopwords = []
                with open('stopwords_en.txt') as file:
                    stopwords = file.read().splitlines()    
                
                # Remove stopwords from all reviews
                stop_ftitle = [token for token in single_ftitle if token not in stopwords]
                stop_fcontent = [token for token in single_fcontent if token not in stopwords]

                # Join the tokenized words with space
                processed_ftitle =  ' '.join(stop_ftitle)
                processed_fcontent = ' '.join(stop_fcontent)

                # Combine the words from title and description
                processed_fInfo = processed_ftitle + ' ' + processed_fcontent

                # Function to vectorize a single review
                def vectorize_text(text, model, vectorizer):
                    words = text.split()
                    tfidf_weights = vectorizer.transform([text]).toarray()[0]
                    feature_names = vectorizer.get_feature_names_out()
                    weighted_vectors = []
                    for word in words:
                        if word in feature_names:
                            try:
                                weighted_vectors.append(model.wv[word] * tfidf_weights[feature_names.tolist().index(word)])
                            except KeyError:
                                pass
                    if len(weighted_vectors) > 0:
                        return sum(weighted_vectors) / len(weighted_vectors)
                    else:
                        return np.zeros(model.vector_size)
                    
                # Load saved FastText and Logistic Regression models and vectorizer
                ft_model = FastText.load('fasttext_model.bin')
                with open('logistic_regression_model.pkl', 'rb') as f:
                    lr_model = pickle.load(f)
                with open('tfidf_vectorizer.pkl', 'rb') as f:
                    tfidf_vectorizer = pickle.load(f)

                # Vectorize the processed and combined review
                vectorized_fInfo = vectorize_text(processed_fInfo, ft_model, tfidf_vectorizer)
                
                # Predict the label of tokenized_data
                y_pred = lr_model.predict(vectorized_fInfo.reshape(1, -1))

                y_pred = y_pred[0]

                
                # Render the classify page with prediction
                return render_template('classify.html', y_pred= y_pred, title=f_title, description=f_content, rating=f_rating, clothing_id =f_id, class_name = f_class)

            # Check if the inputed recommendation is 0 or 1 only
            elif 'submit_review' in request.form:
                if f_recommend not in ['0', '1']:
                    # If inputed recommendation label is not 0 or 1 then display error message
                    message = "The recommendation value must be either 0 (No) or 1 (Yes)!"
                    return render_template('classify.html', title=f_title, description=f_content, rating=f_rating, y_pred=f_recommend, message=message, clothing_id =f_id , class_name = f_class)

                new_id = og_data['ID'].max() + 1

                # Parse the new review into a dictionary
                new_review = { 
                    'Clothing ID': int(f_id),
                    'Age' : '',
                    'Title': f_title,
                    'Review Text': f_content,
                    'Rating': f_rating,
                    'Recommended IND': int(f_recommend),  
                    'Positive Feedback Count':'',
                    'Division Name':'',
                    'Department Name': '',
                    'Class Name':f_class,
                    'Clothes Title': '', 
                    'Clothes Description':'' ,
                    'ID' : new_id
                }
                
                # Append the new review to the original DataFrame
                new_review_df = pd.DataFrame([new_review])
                og_data = pd.concat([og_data, new_review_df], ignore_index=True)

                # Process the new review for the processed 'data' DataFrame
                new_review_processed = {
                    
                    'Clothing ID': f_id,
                    'Age' : '',
                    'Title': f_title,
                    'Review Text': f_content,
                    'Rating': f_rating,
                    'Recommended IND': f_recommend,  
                    'Positive Feedback Count':'',
                    'Division Name':'',
                    'Department Name': '',
                    'Class Name':f_class,
                    'Clothes Title': '', 
                    'Clothes Description':'',
                    'Processed Class Name': stemmer.stem(f_class.lower()),
                    'Processed Clothes Title': stemmer.stem(f_title.lower()),
                    'Processed Clothes Info': stemmer.stem(f_class.lower()) + ' ' + stemmer.stem(f_title.lower()),
                    'ID' : new_id
                }

                processed_review_df = pd.DataFrame([new_review_processed])
                # Add the processed review to 'data' dataframe
                data = pd.concat([data, processed_review_df], ignore_index=True)

                # Display the message if review was submitted
                message = f"Review was successfully submitted!"
                
                
                # Render the classify page with prediction and inputed data
                return render_template('classify.html', title=f_title, description=f_content, rating=f_rating, message= message, clothing_id =f_id, class_name = f_class)
        
        else:
            return render_template('classify.html')

# Define the route for page not found
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Define the route if any error was encountered
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# References
# Dr. Sarwar, T. (2024). Week 10, 11 : Python Flask - Lab [Practical manual]. Canvas@RMIT University. https://rmit.instructure.com

# Jadeja, M. (2022). Jaccard Similarity Made Simple: A Beginner’s Guide to Data Comparison. Medium. https://medium.com/@mayurdhvajsinhjadeja/jaccard-similarity-34e2c15fb524
