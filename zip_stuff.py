import os
import shutil

from misc import load_results

def main():
    results = load_results.get_extra_compression_results("final")
    
    os.mkdir("final_models")

    for model_group in results:
        for result_group in model_group:
            for result_path in result_group:
                shutil.copy(result_path, "final_models")
    
    shutil.make_archive("final_models", "zip", "final_models")
