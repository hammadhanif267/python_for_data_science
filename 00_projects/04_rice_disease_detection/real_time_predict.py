from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model("model/rice_model.h5")
classes = ['Leaf smut', 'Brown spot', 'Bacterial leaf blight']

# Disease descriptions
descriptions = {
    'Leaf smut': "Leaf smut is a fungal disease that causes black spots on leaves, leading to reduced photosynthesis.",
    'Brown spot': "Brown spot is caused by a fungus and results in circular brown lesions on leaves, affecting growth.",
    'Bacterial leaf blight': "This disease is caused by bacteria, leading to water-soaked spots that turn brown."
}

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    
    index = np.argmax(preds[0])
    label = classes[index]
    probability = preds[0][index] * 100  # Convert to percentage
    description = descriptions[label]

    return {
        'label': label,
        'probability': round(probability, 2),
        'description': description
    }