# coin-vision

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

The Coin Detection Project is a computer vision application designed to identify and extract coins from images with precision. Using the YOLOv8 object detection model for initial identication and MobileNetV2 for coin classifaction. The project first detects circular or oval shapes that correspond to coins in various image formats and then match them to one of the 8 coinc classes. It outputs the value of the coins presented in the picture.

+-------------------------------+
|    Input Image with Coins     |
+-------------------------------+
                |
                v
+--------------------------------+
|    YOLOv8: Detect Coins         |
+--------------------------------+
                |
                v
+-------------------------------------------+
|  Filter Detections:                        |
|  - Remove Duplicates                       |
|  - Identify Circular/Oval Shapes           |
+-------------------------------------------+
                |
                v
+-----------------------------------+
|  MobileNetV2: Classify Each Coin  |
+-----------------------------------+
                |
                v
+---------------------------------+
|  Calculate Total Coin Value     |
+---------------------------------+
                |
                v
+----------------------------------+
|   Output the Overall Coin Value  |
+----------------------------------+


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Output of internal training and work
│   ├── processed      <- The final images to be used for training and testing
│   └── raw            <- The original uploaded images, raw
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         coin_vision and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── coin_vision   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes coin_vision a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

