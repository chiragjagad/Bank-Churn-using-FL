import flwr as fl

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_eval=1,
        min_eval_clients=10,
        min_available_clients=10,
    )
    fl.server.start_server("127.0.0.1:8080", config={
                           "num_rounds": 5}, strategy=strategy)
