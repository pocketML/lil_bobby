import requests

URL = "https://mhooge.com/pocketml/save_status.php"
TRANSPONDER_ACTIVE = False

def make_request(url, data):
    if TRANSPONDER_ACTIVE:
        try:
            requests.post(url, data=data)
        except requests.exceptions.RequestException as e:
            print("Error ignored in online_monitor: " + str(e))

def send_train_start(experiment, model_name, task, total_epochs=None):
    data = {
        "experiment": experiment, "model": model_name, "task": task
    }
    if total_epochs is not None:
        data["total_epochs"] = total_epochs

    make_request(URL, data)

def send_train_status(epoch, acc=None):
    data = {"epoch": epoch}
    if acc is not None:
        data["accuracy"] = acc

    make_request(URL, data)
