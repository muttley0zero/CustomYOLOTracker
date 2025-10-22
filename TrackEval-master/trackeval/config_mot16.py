# config_mot16.py

from trackeval.datasets import MotChallenge2DBox
from trackeval.eval import TrackerEval
import trackeval

# Ścieżki do katalogów
GT_FOLDER = r"C:\Users\Bartek\Downloads\TrackEval-master\TrackEval-master\data\gt\mot_challenge"
TRACKERS_FOLDER = r"C:\Users\Bartek\Downloads\TrackEval-master\TrackEval-master\data\trackers\mot_challenge"

# Konfiguracja datasetu
dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dataset_config.GT_FOLDER = GT_FOLDER
dataset_config.TRACKERS_FOLDER = TRACKERS_FOLDER
dataset_config.TRACKERS_TO_EVAL = ['MyTracker']  # nazwa Twojego trackera
dataset_config.CLASSES_TO_EVAL = ['pedestrian']
dataset_config.BENCHMARK = 'MOT16'
dataset_config.SPLIT_TO_EVAL = 'train'
dataset_config.INPUT_AS_ZIP = False
dataset_config.DO_PREPROC = True
dataset_config.TRACKER_SUB_FOLDER = 'data'
dataset_config.OUTPUT_SUB_FOLDER = ''
dataset_config.SEQMAP_FOLDER = None
dataset_config.SEQMAP_FILE = None
dataset_config.PRINT_CONFIG = True

# Tworzymy listę datasetów
dataset_list = [MotChallenge2DBox(dataset_config)]

# Konfiguracja eval
eval_config = trackeval.Evaluator.get_default_eval_config()
eval_config.USE_PARALLEL = False
eval_config.NUM_PARALLEL_CORES = 8
eval_config.BREAK_ON_ERROR = True
eval_config.PRINT_RESULTS = True
eval_config.PLOT_CURVES = True

# Uruchomienie TrackEval
evaluator = TrackerEval(eval_config)
evaluator.eval(dataset_list)
