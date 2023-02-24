"""
Creator: HOCQUET Florian, Landza HOUDI
Date: 30/09/2022
Version: 1.0

Purpose: Manages the project's constants paths.
"""

# IMPORT: utils
import os

"""
ROOT
"""
ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "picchb2022_estimation_bruit")

"""
RESOURCES
"""
RESOURCES_PATH = os.path.join(ROOT_PATH, "resources")

"""
DATASET INFO
"""
DATASETS_INFO_PATH = os.path.join(RESOURCES_PATH, "bdd_description")

TRAIN_INFO_PATH = os.path.join(DATASETS_INFO_PATH, "bdd_train_description.csv")
INFERENCE_INFO_PATH = os.path.join(DATASETS_INFO_PATH, "bdd_inference_description.csv")

"""
TRAIN SAVE
"""
MODEL_SAVE_PATH = os.path.join(RESOURCES_PATH, "models")
TRAIN_RES_PATH = os.path.join(RESOURCES_PATH, "models", "training_results.csv")

"""
INFERENCE SAVE
"""
INFERENCE_SAVE_PATH = os.path.join(RESOURCES_PATH, "inference")
INFERENCE_RES_PATH = os.path.join(RESOURCES_PATH, "inference", "inference_results.csv")

"""
TEST
"""
TEST_PATH = os.path.join(ROOT_PATH, "tests")
TEST_RESOURCE_PATH = os.path.join(TEST_PATH, "resources")

MODEL_TEST_SAVE_PATH = os.path.join(TEST_RESOURCE_PATH, "models")
