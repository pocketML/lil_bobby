import requests

URL = "https://mhooge.com/pocketml/save_status.php"
TRANSPONDER_ACTIVE = False
EXPERIMENT_NAME = None

def make_request(url, json_data):
    if TRANSPONDER_ACTIVE:
        try:
            requests.post(url, json=json_data)
        except requests.exceptions.RequestException as e:
            print("Error ignored in online_monitor: " + str(e))

def send_train_start(experiment, experiment_args, total_epochs=None):
    global EXPERIMENT_NAME
    EXPERIMENT_NAME = experiment
    data = {"experiment": experiment, "new": True}
    if total_epochs is not None:
        data["total_epochs"] = total_epochs

    data["experiment_args"] = experiment_args

    make_request(URL, data)

def send_train_status(epoch, acc=None):
    data = {
        "experiment": EXPERIMENT_NAME, "epoch": epoch, "new": False
    }
    if acc is not None:
        data["accuracy"] = f"{acc:.4f}"

    make_request(URL, data)
