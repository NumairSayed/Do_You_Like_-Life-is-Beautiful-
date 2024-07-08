import streamlit as st
import numpy as np
from PIL import Image
import os

# Trained parameters (including bias as the first entry)
params = [-1.50770104, 0.23125011, -0.00264933, -0.13253112, -0.09890288,
           0.25299765, 0.00288609, -0.02645074, 0.21097317, -0.01242524,
           -0.09264137, 0.08419144, 0.04450287, 0.17011991, -0.07842808, 
          -0.0292802, -0.02885441, -0.01391319, 0.22266702, 1.86526195]


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Prediction function
def predict(X, W):
    z = np.dot(X, W)
    return sigmoid(z)

# Classification function
def classify(X, W, threshold=0.5):
    #X = np.insert(X, 0, 1)  # Add bias term
    return 1 if predict(X, W) > threshold else 0

# List of movies
movies = ["3 Idiots", "Bourne Identity", "Bruce Almighty", "Forest Gump", 
          "How to Lose a Guy in 10 Days", "I Robot", "Independence Day", 
          "La Vita E Bella", "Lord of the Rings", "Oceans 11", "Patriot", 
          "Pearl Harbor", "Pirates", "Pulp Fiction", "Rat Race", "Shrek", 
          "Star Wars", "What Women Want", "When Harry Met Sally"]

# Load cover images
cover_images = {
    "3 Idiots": "Images/3 Idiots.jpg",
    "Bourne Identity": "Images/Bourne Identity.jpg",
    "Bruce Almighty": "Images/Bruce Almighty.jpg",
    "Forest Gump": "Images/Forrest Gump.jpg",
    "How to Lose a Guy in 10 Days": "Images/How to Lose a Guy in 10 Days.jpg",
    "I Robot": "Images/I Robot.jpg",
    "Independence Day": "Images/Independence Day.jpg",
    "La Vita E Bella": "Images/La Vita E Bella.jpg",
    "Lord of the Rings": "Images/Lord of the Rings.jpg",
    "Oceans 11": "Images/Oceans 11.jpg",
    "Patriot": "Images/Patriot.jpg",
    "Pearl Harbor": "Images/Pearl Harbor.jpg",
    "Pirates": "Images/Pirates.jpg",
    "Pulp Fiction": "/Images/Pulp Fiction.jpg",
    "Rat Race": "Images/Rat Race.jpg",
    "Shrek": "Images/Shrek.jpg",
    "Star Wars": "Images/Star Wars.jpg",
    "What Women Want": "Images/What Women Want.jpg",
    "When Harry Met Sally": "Images/When Harry Met Sally.jpg"
}

# Streamlit app
st.title("Do You like -Life is Beautiful-?")
st.write("Select the movies you have watched and liked:")

selected_movies = []

# Show cover images with checkboxes for selection
cols = st.columns(5)
for i, movie in enumerate(movies):
    with cols[i % 5]:
        image_path = cover_images[movie]
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=movie, use_column_width=True)
            if st.checkbox(f"Select {movie}", key=movie):
                selected_movies.append(movie)
        else:
            st.write(f"Cover image for {movie} not found.")

# Predict if user will like "Life is Beautiful"
if st.button("Predict"):
    X = [1 if movie in selected_movies else 0 for movie in movies]
    X = np.insert(X, 0, 1)  # Add bias term here
    like_life_is_beautiful,temp = classify(X, params),predict(X, params)

    if like_life_is_beautiful:
        st.success("You will likely enjoy 'Life is Beautiful'! \n With a probability of :{temp}")
    else:
        st.error("You may not enjoy 'Life is Beautiful' \n With a probability of :{temp}.")

# Explanation of the neural network
st.header("Neural Network Structure")
st.write("""
This neural network is a simple logistic regression model trained to predict if a user will like the movie "Life is Beautiful" based on their preferences for other movies. 

### Structure:
- **Input Layer**: 19 neurons, one for each movie.
- **Output Layer**: 1 neuron with a sigmoid activation function.

### Diagram:
""")

# Neural network diagram
st.image("Images/neural_network.png", caption="Neural Network Structure")

st.write("### Calculations:")
st.latex(r"\sigma(x) = \frac{1}{1 + e^{-x}}")
st.latex(r"\hat{y} = \sigma(W \cdot X + b)")
st.latex(r"\text{If } \hat{y} > 0.5 \text{, predict 1 (like)}")

st.write("### Parameters:")
st.write(params)

