from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model = load_model("model/rice_model.h5")

test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory('preprocessed_data/test', target_size=(128, 128), class_mode='categorical', shuffle=False)

preds = model.predict(test_data)
y_pred = np.argmax(preds, axis=1)
y_true = test_data.classes

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))