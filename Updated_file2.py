import streamlit as st
import pandas as pd
import numpy as np
import re
import base64
from io import StringIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import pickle


st.title("Cornershop SKU Classifier v2.1.1")

uploaded_file = st.file_uploader("Choose a csv file")
if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)
    
    
    count_vec = CountVectorizer()
    bow = count_vec.fit_transform(df['item_name'].values.astype('U'))
    dense_bow = bow.todense()
    bow = np.array(dense_bow)
    X = bow
    
    filename = 'finalized_model_all_items.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    
    result = loaded_model.predict(X)
    output = pd.DataFrame(result)
    output['Input'] = df['item_name']
    output.rename(columns = {0:'Output'}, inplace = True)
    output = output[['Input','Output']]
    output[['category_id', 'parent_id','Sub-Category-en','Category-en']] = output['Output'].str.split('__', 3, expand=True)
    output = output[['Input', 'category_id', 'parent_id', 'Sub-Category-en', 'Category-en']]
    
    probs = pd.DataFrame(loaded_model.predict_proba(X), columns=loaded_model.classes_)
    probs['Max'] = probs.max(axis=1)
    s1 = pd.Series(probs['Max'], name="Max_prob")
    result = pd.concat([output, s1], axis=1)
    
    
    
    
    st.write(result)
    
    csv = df_final.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    


    st.markdown('### **⬇️ Download output CSV File **')
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)
