import streamlit as st
from PIL import Image
import numpy as np
from potato_classification import classify_image

st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon = ":potato:"
)
kaggle_url = "https://www.kaggle.com/datasets/mukaffimoin/potato-diseases-datasets"


page = st.radio("Choose a page:",
		["Demo", "Explanation"], captions = ["Small demo app.","Written explanation on the creation of the model."])

if page == "Demo":
	st.write("## Potato Disease Classifier.")
	st.write("Small demo app for classification of common potato diseases.")
	st.write("The model is trained to detect Black Scruf, Black Leg, Common Scab, Dry Rot, Healthy, Misc Diseases, and Pink Rot")
	st.write("[Dataset](%s)"%kaggle_url)
	uploaded_image = st.file_uploader(label="Upload a .jpg or a .png of a potato to get a classification:", type= ["jpg","png"], accept_multiple_files=False)


	if uploaded_image:
		image_placeholder = st.empty()
		prediction_placeholder = st.empty()
	else:
		st.write("Upload an image to get a prediction.")

	if uploaded_image:
			image = Image.open(uploaded_image)
			image_placeholder.image(image)
    
			with st.spinner("Checking potatoes..."):
					prediction= classify_image(image)  # Assuming classify_image is your prediction function
			
			if prediction == "Healthy":
				prediction_placeholder.success(f"Prediction: {prediction}")
			else:
				prediction_placeholder.error(f"Disease Detected: {prediction}")
				if prediction == "Black Scruf" or "Dry Rot" or "Pink Rot":
					st.write("Cause of disease is **Fungus**.")
				elif prediction =="Black Leg" or "Common Scab":
					st.write("Cause of disease is **Bacteria**.")
				else:
					pass


##Explanation page
elif page == "Explanation":
	st.write("## The process I took to create the potato disease classifer model.")
	st.write("**Step 1:** Obtain dataset.")
	
	st.write("The data for this project was taken from [Kaggle.](%s)"%kaggle_url)

	st.write("**Step 2:** Analyse and split the data")
	st.write("The dataset available on Kaggle is quite limited, consisting of only *451* images. This small dataset size is insufficient for training a model effectively. To address this issue, I decided to upload the dataset to Roboflow, a platform that offers various features for generating more images. With Roboflow, I was able to create additional images with options for adding blur and exposure offsets. This approach helps the model to better handle camera-related issues that may arise in real-world scenarios. Using Roboflow also allows me to split the data into training, test, and validation folders before redownloading it.")

	st.write("**Step 3:** Create an environment for training.")
	st.write("The next step was to create a virtual Python environment and use pip to install Ultralytics.")

	st.code("python -m venv potato_venv", language="python")
	st.code("pip install Ultralytics", language="python")

	st.write("**Step 4:** Train the model.")
	st.write("This step involves using a YOLO classification model and pointing it towards the folder containing the split data. ")
	code = """
	from ultralytics import YOLO
	import numpy as np

	model = YOLO("yolov8n-cls.pt")

	path = 'C:\\the\\absoulte\\path\\to\\the\\data'
	model.train(data=path, epochs=50, imgsz=128)
	 """
	st.code(code, language="python")

	st.write("**Step 5:** Review the models performance.")
	st.write("Once the model is trained with YOLO I can view the performance graphs during training.")
	col1, col2 = st.columns(2)
	conf = Image.open("E:\\MachineLearning\\Portfolio\\STREAMLIT\\images\\potato_explain\\confusion_matrix.png")
	graphs = Image.open("E:\\MachineLearning\\Portfolio\\STREAMLIT\\images\\potato_explain\\results.png")
	with col1:
		st.image(conf,caption="Confusion Matrix")
	with col2:
		st.image(graphs,caption="Performance over epochs")

	st.write("The model seems to perform ok with a clearly defined diagonal line in the confusion matrix but still makes some prediction errors. This could be combated with a larger training set however using Roboflow to expand the dataset further is a premium feature and the model performs within the bounds of my use case of a practice model with no plans for real-world uses.")

	st.write("**Step 6:** Deploy the model to streamlit.")
	st.write("The model is available as a demo and can be selected with the radio buttons above.")