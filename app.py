import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('handwritten_digit_model.h5')

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwritten Digit Recognizer")
        
        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack()
        
        self.image1 = Image.new("L", (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image1)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.btn_predict = tk.Button(master, text="Predict", command=self.predict_digit)
        self.btn_predict.pack()
        
        self.btn_clear = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.btn_clear.pack()
        
        self.label = tk.Label(master, text="Draw a digit and click Predict", font=("Helvetica", 16))
        self.label.pack()

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=20)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill='white')
        self.label.config(text="Draw a digit and click Predict")

    def predict_digit(self):
        # Resize and invert the image
        img = self.image1.resize((28, 28))
        img = ImageOps.invert(img)
        
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)
        
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        self.label.config(text=f"Predicted Digit: {digit} (Confidence: {confidence*100:.2f}%)")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
