import unittest
import os
import glob
from denovonet.training.train import training_pipeline


PATH_TO_FIXTURES = "fixtures"
TEMP_DIR = "temp"


class TestTrainModelPytorch(unittest.TestCase):
    def tearDown(self):
        files = glob.glob(f"{TEMP_DIR}/*")
        for f in files:
            os.remove(f)

    def test_training_pipeline(self):
        batch_size = 2
        images_folder = os.path.join(PATH_TO_FIXTURES, "test_images", "substitution")
        epochs = 2
        workdir = "temp"
        outname = "test_model"
        training_pipeline(
            batch_size=batch_size,
            images_folder=images_folder,
            epochs=epochs,
            workdir=workdir,
            outname=outname,
        )
