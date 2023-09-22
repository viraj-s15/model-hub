import keras_ocr

pipeline = keras_ocr.pipeline.Pipeline()

images = [keras_ocr.tools.read(img) for img in ["directory"]]

# generate text predictions from the images
prediction_groups = pipeline.recognize(images)

predicted_image = prediction_groups[1]
for text, box in predicted_image:
    print(text)
