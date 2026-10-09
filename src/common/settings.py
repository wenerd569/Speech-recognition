
import json
import os

from exceptions.exceptions import WordInManyClassesException

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "path_conf.conf")





# class PathSettings:

#     def __init__(self, settings_file=DEFAULT_SETTINGS_FILE):
#         config = configparser.ConfigParser()
#         loaded_files = config.read(settings_file)
#         if not loaded_files:
#             raise FileNotFoundError(f"Settings file not found: {settings_file}")

#         self._dir_current = self._resolve_path(config.get("paths", "DIR_CURRENT"))
#         self._dir_to_upload_from = self._resolve_path(
#             config.get("paths", "DIR_TO_UPLOAD_FROM")
#         )
#         self._dir_to_save_to = self._resolve_path(config.get("paths", "DIR_TO_SAVE_TO"))
#         self._df_path = self._resolve_path(config.get("paths", "DF_PATH"))



DEFAULT_TRAIN_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "train_settings.json")

class TrainSettings:

    def __init__(self, train_settings_path=DEFAULT_TRAIN_SETTINGS_FILE):

        with open(train_settings_path, encoding="utf-8") as settings_file:
            config = json.load(settings_file)

        base_settings = config["base_settings"]
        
        self._batch_size = int(base_settings["batch_size"])
        self._model_saving_path = self._resolve_path(
            base_settings["model_saving_path"]
        )
        self._place_to_store_dataset = base_settings["place_to_store_dataset"]
        self._num_classes = len(config["classes"])
        self._labels_to_index = self._resolve_labels(config["classes"])

        self._dir_to_upload_from = self._resolve_path(base_settings["dir_to_upload_from"])
        self._dir_to_save_to = self._resolve_path(base_settings["dir_to_save_to"])
        self._df_path = self._resolve_path(base_settings["df_path"])

        self._epoch_count = base_settings["epoch_count"]
        self._parts = config["parts"]

    def _resolve_path(self, path):
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        return os.path.abspath(path)

    def _resolve_labels(self, arr_arr):
        res = {}
        for i, arr in enumerate(arr_arr):
            for el in arr:
                if el in res:
                    raise WordInManyClassesException(el)
                res[el] = i
        return res

    def get_batch_size(self):
        return self._batch_size

    def get_model_saving_path(self):
        return self._model_saving_path

    def get_place_to_store_dataset(self):
        return self._place_to_store_dataset

    def get_labels_to_index(self):
        return self._labels_to_index
    
    def get_num_classes(self):
        return self._num_classes

    def get_dir_to_upload_from(self):
        return self._dir_to_upload_from

    def get_dir_to_save_to(self):
        return self._dir_to_save_to

    def get_df_path(self):
        return self._df_path
    
    def get_epoch_count(self):
        return self._epoch_count

    def get_parts(self):
        return self._parts