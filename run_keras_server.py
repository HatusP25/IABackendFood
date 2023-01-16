# import the necessary packages
import keras
from flask_cors import cross_origin
from keras.applications import ResNet50
# from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
from keras.applications.vgg16 import preprocess_input
from keras.utils import img_to_array

app = flask.Flask(__name__)
model = None
labels = ['Guaguas de pan', 'Hamburguesa', 'Pizza', 'Seco de Pollo', 'Sushi', 'Tacos', 'bolon', 'cazuela', 'ceviche', 'empanadas', 'encebollado', 'encocado de pescado', 'fanesca', 'lasagna', 'tigrillo']
caloriesDict = {
    'Guaguas de pan': 372.4,
    'Hamburguesa': 295,
    'Pizza': 266,
    'Seco de Pollo': 542.8,
    'Sushi': 350,
    'Tacos': 226,
    'bolon': 331,
    'cazuela': 329,
    'ceviche': 173,
    'empanadas': 267,
    'encebollado': 245,
    'encocado de pescado': 228.2,
    'fanesca': 193,
    'lasagna': 336,
    'tigrillo': 136
}


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = keras.models.load_model("FoodRecognitionModelV9.h5")



def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # return the processed image
    return image


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))


            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))


            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            print(np.argmax(preds))
            result = labels[np.argmax(preds)]
            data["prediction"] = []

            # loop over the results and add them to the list of
            # returned predictions
            r = {"label": result, "probability": float(preds[0][np.argmax(preds)]), "calories": caloriesDict[result]}
            data["prediction"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()
