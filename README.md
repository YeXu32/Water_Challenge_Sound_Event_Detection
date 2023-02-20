# Water_Challenge_Sound_Event_Detection
Hackathon Challenge to detect water faucet on and off time in given audios 

## Challenge: 
TBC

## Reference: 
Repository built from https://github.com/YashNita/sound-event-detection-winning-method \
Sharath Adavanne, Pasi Pertila and Tuomas Virtanen, "Sound event detection using spatial features and convolutional recurrent neural network" in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2017) 

## Prerequisites
The Python script is written and tested in 3.8.8 version. \
You can install the requirements by running the following line:
```
pip install -r requirements.txt
```

## Content:
This repository consists of five Python scripts.
1. The feature.py script, extracts the features, labels, and normalizes the training and test split features.
2. The sed.py script, loads the normalized features, and traines the SEDnet. The training stops when the error rate metric in one second segment (http://tut-arg.github.io/sed_eval/) stops improving.
3. The metrics.py script, implements the core metrics from sound event detection evaluation module http://tut-arg.github.io/sed_eval/
4. The utils.py script has some utility functions.
5. The postprocessing.py script converts the predicted value into timestamps 


## Model Structure: 
<img src="https://github.com/YeXu32/Water_Challenge_Sound_Event_Detection/blob/main/CRNN_SED.jpg" alt="Alt text" title="Optional title">
