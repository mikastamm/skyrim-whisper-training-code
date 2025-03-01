#!/bin/bash
source venv_training/bin/activate
python 3.2-train.py
python 4.1-eval-test-set.py
python 4.2-eval-training-data-efficiency.py
python 4.3-eval-common-voice.py