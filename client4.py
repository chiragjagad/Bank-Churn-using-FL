import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
import flwr as fl
import pickle
import time

if __name__ == "__main__":
    clientID = 4
    model = Sequential(
        [
            Dense(24, input_shape=(33,), activation=LeakyReLU(alpha=0.01)),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dropout(0.2),
            Dense(8, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    with open("xtrain.pkl", "rb") as f:
        x_train = pickle.load(f)
        x_train = x_train[1360*2*(clientID - 1): 1360*2*clientID]
    with open("xtest.pkl", "rb") as f:
        x_test = pickle.load(f)
    with open("ytrain.pkl", "rb") as f:
        y_train = pickle.load(f)
        y_train = y_train[1360*2*(clientID - 1): 1360*2*clientID]
    with open("ytest.pkl", "rb") as f:
        y_test = pickle.load(f)
    loss_arr = []
    acc_arr = []
    time_arr = []
    # Define Flower client

    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            start = time.process_time()
            history = model.fit(x_train, y_train, epochs=100,
                                batch_size=256, verbose=0)
            end = time.process_time()
            time_arr.append(end - start)
            acc_arr.append(history.history["accuracy"][0])
            loss_arr.append(history.history["loss"][0])
            results = {
                "loss": history.history["loss"][0],
                "accuracy": history.history["accuracy"][0],
            }
            print(time_arr)
            print(acc_arr)
            print(loss_arr)
            return model.get_weights(), len(x_train), results

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("127.0.0.1:8080", client=CifarClient())
