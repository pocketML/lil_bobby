from datetime import datetime
import json

from analysis.parameters import get_theoretical_size
from compression.distillation import models as distill_models
from misc import load_results

result_groups = load_results.get_distillation_results()

for result_group in result_groups:
    for result_folder in result_group:
        try:
            with open(f"{result_folder}/metrics.json", "r", encoding="utf-8") as fp:
                name = result_folder.replace("\\", "/").split("/")[-1]
                name_split = name.split("_")
                arch = name_split[0]
                if arch == "embffn":
                    arch = "emb-ffn"
                task = name_split[1]
                if task == "sst":
                    task = "sst-2"

                metrics_data = json.load(fp)

                model = distill_models.load_student(task, arch, use_gpu=False, model_name=name)
                if "theoretical_size" not in metrics_data:
                    nonzero_params, theoretical_size = get_theoretical_size(model)
                    time_now = datetime.now()
                    metrics_data["nonzero_params"] = {
                        "steps": [0],
                        "timestamp": [time_now.strftime("%Y-%m-%dT%H:%M%S.%f")],
                        "values": [nonzero_params]
                    }
                    metrics_data["theoretical_size"] = {
                        "steps": [0],
                        "timestamp": [time_now.strftime("%Y-%m-%dT%H:%M%S.%f")],
                        "values": [theoretical_size]
                    }

        except (IOError, KeyError):
            pass
