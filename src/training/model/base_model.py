from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass