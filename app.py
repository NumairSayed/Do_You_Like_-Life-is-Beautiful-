import streamlit as st
import numpy as np
from PIL import Image
import os

# Trained parameters (including bias as the first entry)
params = [-1.53026782e+00,  2.27224786e-01, -6.46238237e-04, -1.32040296e-01,
          -9.91159047e-02,  2.52158756e-01,  2.86539638e-03, -2.56549917e-02,
           2.07797876e-01, -1.27764689e-02, -9.23999152e-02,  8.29330515e-02,
           4.40164550e-02,  1.69074113e-01, -7.79385320e-02, -2.88810124e-02,
          -2.79895281e-02, -1.34292568e-02,  2.21425727e-01,  1.86446175e+00,
           5.83075325e-02]

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Prediction function
def predict(X, W):
    z = np.dot(X, W)
    return sigmoid(z)

# Classification function
def classify(X, W, threshold=0.5):
    X = np.insert(X, 0, 1)  # Add bias term
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
    like_life_is_beautiful = classify(X, params)

    if like_life_is_beautiful:
        st.success("You will likely enjoy 'Life is Beautiful'! \n With a probability of :{predict(X,params)}")
    else:
        st.error("You may not enjoy 'Life is Beautiful' \n With a probability of :{predict(X,params)}.")

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

