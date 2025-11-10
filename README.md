# Dice Detector – OpenCV + DBSCAN

A real-time dice recognition project that detects pips (dots) on dice using
**OpenCV SimpleBlobDetector** and clusters them into individual dice using
**DBSCAN**. The system automatically counts pips, groups them, and displays the
value of each die and the total.


## Features

- Detects pips (dice dots) using blob detection  
- Groups pips into dice using **DBSCAN clustering**  
- Draws bounding boxes around each die  
- Displays the value of each die and the total  
- Works with video files or live camera streams  
- Easy to adjust blob detector parameters  


## Installation

```bash
pip install opencv-python numpy scikit-learn
