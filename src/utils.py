import os
import sys

import numpy as np
import pandas as pd
import dill


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and pass the file object to dill.dump
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # Correct: Pass file_obj and obj

    except Exception as e:
        raise CustomException(e, sys)
