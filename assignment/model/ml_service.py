import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
# https://redis-py.readthedocs.io/en/stable/connections.html#
db = redis.Redis(
    host= settings.REDIS_IP,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID)

# Load your ML model and assign to variable `model`
# See https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing
# for more information about how to use this model.
model = ResNet50(include_top=True, weights="imagenet")

def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    class_name = None
    pred_probability = None

    # Now it's time to load the image
    # We will use `image` module from tensorflow.keras
    # We also need to change the image to the input image size
    # the Resnet50 model is expecting, in this case (224, 224).
    img = image.load_img(
        os.path.join(settings.UPLOAD_FOLDER, image_name),
        target_size=(224, 224)
    )

    # Convert the PIL image to a Numpy
    # array before sending it to the model
    image_array = image.img_to_array(img)
 
    # Also add an extra dimension to this array
    # because our model is expecting as input a batch of images.
    # In this particular case, we will have a batch with a single
    # image inside
    image_batch = np.expand_dims(image_array, axis=0)
    
    # Now we scale pixels values
    image_batch = preprocess_input(image_batch)

    # Run model on batch of images (only one)
    predictions = model.predict(image_batch)

    # Get the predicted label with the highest probability
    top_prediction = decode_predictions(predictions, top=1)

    # Get the first result
    # top_prediction format: 3-dimension list
    # [[('n02108551', 'Tibetan_mastiff', 0.9666902)]]
    class_name = top_prediction[0][0][1]
    pred_probability = top_prediction[0][0][2]

    # Round the probability to 4 places (just to pass the test 😉)
    return class_name, round(pred_probability, 4)


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
        # Hint: You should be able to successfully implement the communication
        #       code with Redis making use of functions `brpop()` and `set()`.
        
        # 1
        # For reading messages from the queue we can use `brpop()`.
        # It will remove and return the last element of the list.
        # If the list is empty, it will block the connection,
        # waiting for some new element to appear.
        # https://redis.io/docs/latest/commands/brpop/
        queue_name, msg_json = db.brpop(settings.REDIS_QUEUE)

        # 2
        # Convert json to dict
        msg = json.loads(msg_json)

        # Call predict function, passing in the image name
        class_name, pred_probability = predict(msg["image_name"])

        # 3
        prediction_dict = {
            "prediction": class_name,
            "score": str(pred_probability),     #Convert to string to serialize
        }
        prediction_json = json.dumps(prediction_dict)

        # 4
        msg_id = msg["id"]
        db.set(msg_id, prediction_json)

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
