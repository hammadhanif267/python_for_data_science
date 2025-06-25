import streamlit as st
import pandas as pd
import numpy as np
# Adding a title of your app
st.title('My First Testing App')

# Adding simple text
st.write('Here is a simple text')

# User input
number = st.slider('Select a number', 0, 100, 50)  # 50 is the default value

# Print the text of number
st.write(f'Your selected number: {number}')

# Adding a button
if st.button('Greetings'):
    st.write('Hi, hello there')
else:
    st.write('Click the Greetings button to get a greeting')

# Add radio button with options
genre = st.radio(
    "What's your favorite movie genre?",
    ('Comedy', 'Drama', 'Documentary'))
# print the text of the selected genre
st.write(f'You selected: {genre}')

# # add a drop down list with options
# option = st.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home Phone', 'Mobile Phone'))
# # print the text of the selected option
# st.write(f'You selected: {option}')

# add a drop down list with options on the left sidebar
option = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home Phone', 'Mobile Phone'))
# print the text of the selected option
st.sidebar.write(f'You selected: {option}')

# add your whatsapp number
st.sidebar.text_input('Enter your WhatsApp number')

# add a file uploader
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# create a line plot
# Plotting
data = pd.DataFrame({
  'first column': list(range(1, 11)),
  'second column': np.arange(number, number + 10)
})
st.line_chart(data)