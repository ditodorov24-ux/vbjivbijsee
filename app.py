1 import streamlit as st
2 import numpy as np
3 from PIL import Image
4 import joblib
5 import requests
6 import io
7
8
9
st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="")
10
st.title(" Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and AI will try to recognize it.")
11
# Simple model loading with fallback
12
13 @st.cache_resource
14
def load_model():
15
try:
16
# Try to load pre-trained model
17
# Using sklearn's built-in digits dataset
18
from sklearn.datasets import load_digits
19
from sklearn.neural_network import MLPClassifier
20
from sklearn.model_selection import train_test_split
21
22
digits load_digits()
23
X = digits.images.reshape((len(digits.images), -1)) / 16.0
24
= digits.target
25
X_train, _,y_train, = train_test_split(X, y, test_size=0.2, random_state=42)
26
model MLPClassifier(
28
27
hidden_layer_sizes=(100,),
29
max_iter=100,
30
random state=42
31 )
32
model.fit(X_train, y_train)
33
return model
34 except Exception as e:
35
st.error(f"Model loading error: {e}")
36
return None
37
38 model load_model()
39
40 if model is None:
41 st.warning("Could not load model. Using fallback recognition.")
42 else:
43 st.success("Model loaded successfully!")
44
45 # File uploader
46 uploaded_file st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
47
48 if uploaded_file is not None:
49
# Display the uploaded image
50 image Image.open(uploaded_file)
51
st.image(image, caption="Uploaded Image', use_column_width=True)
53 # Process the image
54 try:
55
# Convert to grayscale and resize to 8x8
57
img_gray image.convert('L')
58
img_resized = img_gray.resize((8, 8))
59
# Convert to numpy array and invert if needed
60
img_array = np.array(img_resized)
61
62
63
# If background is dark, invert
64
if np.mean(img_array) > 128:
65
img_array = 255 img_array
66
# Normalize like the training data
67
img_array = img_array / 16.0
img_flat
img_array.flatten().reshape(1, -1)
68
69
if model is not None:
70
# Make prediction
71
prediction model.predict(img_flat) [0]
72
st.write(f"## Prediction: **{prediction}**")
73
74
75
76
# Show probabilities
77
probs model.predict_proba(img_flat) [0]
78
st.write("### Probabilities:")
79
st.write(f"Digit {i}: {prob:.2%}")
80
else:
81
# Fallback: simple threshold-based recognition
82
st.write("## Using fallback recognition").
83
# Simple heuristic based on pixel intensity
84
digit_guess = np.argmax(np.sum(img_array.reshape (8, 8), axis=0)) % 10
85
st.write(f"Estimated digit: **{digit_guess}**")
86
except Exception as e:
87
st.error(f"Error processing image: {e}")
88
89
90
# Instructions
91
st.sidebar.header("Instructions")
92
st.sidebar.write("""
93
1. Upload an image of a handwritten digit (0-9)
94
2. The image will be resized to 8x8 pixels
95
3. AI model will predict the digit
96
4. For best results:
97
White background
98
Black digit
99
Centered digit
100
Minimal noise
101
")
