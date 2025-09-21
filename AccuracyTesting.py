import pickle
import matplotlib.pyplot as plt

f = open('training_history_v2.pkl', 'rb')
history = pickle.load(f)
f.close()

# Plot for accuracy
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy Plot')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'])
plt.show()

# Plot for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss Plot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'])
plt.show()