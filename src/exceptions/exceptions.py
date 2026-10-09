

class WordInManyClassesException(Exception):

    def __init__(self, label):
        self.label = label
    
    def __str__(self):
        return f"Слово {self.label} повторилось в разных классах"


class WrongDatasetSettingsException(Exception):
    pass