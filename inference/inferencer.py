from dataset.dataset_features import create_dataloader, DatasetType


class Inferencer:
    """
    Class to manage inference process
    """
    def __init__(self):
        """
        """
        self.test_dataloader=create_dataloader(type=DatasetType.TEST, shuffle=False, batch_size=16)
        pass
