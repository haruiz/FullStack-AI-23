import typing
from pathlib import Path
import numpy as np
from enum import Enum

class Framework(Enum):
    tensorflow = 'tensorflow'
    sklearn = 'sklearn'
    pytorch = 'pytorch'


class ModelLoader(object):
    def __init__(self, 
                  path: typing.Union[str, Path], 
                  name: str, 
                  version: float = 1.0,
                  framework: Framework = Framework.tensorflow, labels: typing.List[str] = None):
        self.path = path
        self.name = name
        self.version = version
        self.framework = framework
        self.labels = labels

        if self.framework == Framework.tensorflow:
            self.model = self.__load_tensorflow_model()
        elif self.framework == Framework.sklearn:
            self.model = self.__load_sklearn_model()
        else:
            raise NotImplementedError(f'Backend {self.framework} is not supported.')


    def __load_tensorflow_model(self):
        """"
        Load tensorflow model from path
        """
        import tensorflow as tf
        model = tf.keras.models.load_model(self.path)
        return model
    

    def __load_sklearn_model(self):
        """"
        Load sklearn model from path
        """
        import pickle
        with open(self.path, "rb") as f:
            return pickle.load(f)
        

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict data using model
        """
        predictions =  self.model.predict(data)
        predictions = predictions.tolist()
        if self.labels:
            if self.framework == Framework.sklearn:
                predictions = [self.labels[label_idx] for label_idx in predictions]
            elif self.framework == Framework.tensorflow:
                predictions = [self.labels[np.argmax(prediction)] for prediction in predictions]
        return predictions
        



if __name__ == "__main__":
    

    features = np.array([
        [5.1, 3.5, 1.4, .2],
        [5.1, 3.5, 1.4, .2]
    ])

    framework = Framework.tensorflow
    models_path = {
        'tensorflow': 'models/tf/iris_model',
        'sklearn': 'models/sklearn/iris_model.pk'
    }
    

    model = ModelLoader(
        path=models_path[framework.value], 
        name='iris_model', 
        version=1.0, 
        framework=framework, 
        labels=['setosa', 'versicolor', 'virginica']
    )
    
    
    prediction = model.predict(features)
    print(prediction)

    
        
        

    

    