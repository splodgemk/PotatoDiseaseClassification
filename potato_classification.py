from ultralytics import YOLO

model = YOLO("best.pt")

#pot = ("C:\\Users\\Max\\Documents\\leaf.jpg")

class_names = ['Black Scruf', 'Black Leg', 'Common Scab', 'Dry Rot',
"Healthy","Misc Diseases","Pink Rot"]

# Function to classify an image and return the most confident class name
def classify_image(image):
    results = model.predict(image)
    probs = results[0].probs  # Accessing the first result's Probs object
    
    top_class_index = probs.top1  # Directly access the top1 attribute, which is an int
    top_class_name = class_names[top_class_index]  # Use the index to get the class name from class_names list
    top_class_probs = probs.top1
    return top_class_name
