from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path, PurePosixPath
import posixpath
from google.cloud import storage
from torch import save
import io

class DataOrchestrator(ABC):

    @abstractmethod
    def create_experiment_dir(self):
        pass

    @abstractmethod
    def create_model_dir(self, seed: int):
        pass

    @abstractmethod
    def get_model_dir(self, seed: int):
        pass

    @abstractmethod
    def get_model_path(self, model_dir, model_hash):
        pass

    @abstractmethod
    def delete_last_model(self, model_dir_path: str):
        pass

    @abstractmethod
    def list_dir(self, path: str):
        pass

    @abstractmethod
    def extract_folder_name(self, path):
        pass

    @abstractmethod
    def rglob(self, path, suffix):
        pass

    @abstractmethod
    def isfile(self, path):
        pass

    @abstractmethod
    def save_model(self, checkpoint, model_path):
        pass


    @abstractmethod
    def save_df(self, df, path):
        pass


class OsOrchestrator(DataOrchestrator):

    def __init__(self, save_path: str, experiment_hash: str, report_name: str):
        super().__init__()
        self.experiment_dir_path = os.path.join(save_path, experiment_hash)
        self.report_path = os.path.join(save_path, experiment_hash, report_name)


    def create_experiment_dir(self):        
        if not os.path.isdir(self.experiment_dir_path):
            os.makedirs(self.experiment_dir_path)
        else:
            logging.info(f"Experiment directory already exists. Skipping directory creation.")


    def get_model_dir(self, seed: int):
        return os.path.join(self.experiment_dir_path, str(seed))
    

    def get_model_path(self, model_dir, model_hash):
        return os.path.join(model_dir, f"{model_hash}.pt")


    def create_model_dir(self, seed: int):
        model_dir_path = self.get_model_dir(seed)
        
        if not os.path.isdir(model_dir_path):
            os.makedirs(model_dir_path)
        else:
            logging.info(f"Directory for seed: {str(seed)} already exists. Skipping directory creation.")

        return model_dir_path


    def delete_last_model(self, model_dir_path: str):
        '''deletes all files ending with .pt'''

        for file in Path(model_dir_path).glob("*.pt"):
            file.unlink()


    def list_dir(self, path: str):
        return os.list_dir(path)


    def extract_folder_name(self, path: str):
        return Path(path).parent.name


    def rglob(self, path, suffix):
        return [path.rglob(f"*.{suffix}")]


    def isfile(self, path):
        return os.path.isfile(path)


    def save_model(self, checkpoint, model_path):
        save(checkpoint, model_path)
        logging.info(f"Saved model to checkpoint: {model_path}")
    

    def save_df(self, df, path):
        df.to_excel(path, index=False)
        

class GcpOrchestrator(DataOrchestrator):

    def __init__(self, save_path: str, experiment_hash: str, project_name: str, bucket_name: str, report_name: str):
        super().__init__()
        self.client = storage.Client(project=project_name)
        self.bucket = self.client.bucket(bucket_name)
        self.experiment_dir_path = posixpath.join(save_path, experiment_hash).rstrip("/") + "/"
        self.report_path = posixpath.join(save_path, experiment_hash, report_name).rstrip("/") + "/"


    def create_experiment_dir(self):        

        if not self._dir_exists(self.experiment_dir_path):
            self._create_dir(self.experiment_dir_path)
        else:
            logging.info(f"Experiment directory already exists. Skipping directory creation.")


    def get_model_dir(self, seed: int):
        return posixpath.join(self.experiment_dir_path, str(seed)).rstrip("/") + "/"


    def get_model_path(self, model_dir, model_hash):
        return posixpath.join(model_dir, f"{model_hash}.pt")


    def create_model_dir(self, seed: int):
        model_dir_path = self.get_model_dir(seed)

        if not self._dir_exists(model_dir_path):
            self._create_dir(model_dir_path)
        else:
            logging.info(f"Directory for seed: {str(seed)} already exists. Skipping directory creation.")

        return model_dir_path


    def delete_last_model(self, model_dir_path):
        '''deletes all files ending with .pt'''


        print("Deleting last model")
        print(f"model_dir_path: {model_dir_path}")

        blobs = self.list_dir(model_dir_path + "/")
        for blob in blobs:
            if blob.name.endswith(".pt"):
                print(f"Deleting: {blob.name}")
                blob.delete()
    

    def _dir_exists(self, path: str):
        blob = self.bucket.blob(path.rstrip("/") + "/")
        return blob.exists()
    

    def _create_dir(self, path: str):
        blob = self.bucket.blob(path)
        blob.upload_from_string("")


    def list_dir(self, path: str):
        return self.bucket.list_blobs(prefix=path)


    def extract_folder_name(self, path: str):
        return PurePosixPath(path).parent.name


    def rglob(self, path, suffix):
        return [blob.name for blob in self.bucket.list_blobs(prefix=path) if blob.name.endswith(suffix)]


    def isfile(self, path):
        blob = self.bucket.blob(path)
        return blob.exists()


    def save_model(self, checkpoint, model_path):
        blob = self.bucket.blob(model_path)
        buffer = io.BytesIO()
        save(checkpoint, buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer)


    def save_df(self, df, path):
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        self.bucket.blob(path).upload_from_file(buffer)


def get_orchestrator(env: str, save_path: str, experiment_hash: str, project_name: str, bucket_name: str, report_name: str):

    if env == "os":
        orchestrator = OsOrchestrator(save_path, experiment_hash, report_name)
    elif env == "gcp":

        if project_name is None and bucket_name is None:
            raise Exception(f"The env was indicated as {env} but either project_name or bucket_name param seem to be missing.")

        orchestrator = GcpOrchestrator(save_path, experiment_hash, project_name, bucket_name, report_name)
    else:
        raise Exception(f"Indicated env {env} was not recognized.")

    return orchestrator