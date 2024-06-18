# Guide to Get This Working on Your Machine

## Step 1: Download PyCharm and Create a Project

1. Download and install [PyCharm](https://www.jetbrains.com/pycharm/download/).
2. Open PyCharm and create a new project.

## Step 2: Import All Files

1. Place all your project files (`run.py`, `tweakAI.py`, `main.py`, and any other necessary files) into the project directory.

## Step 3: Set Up the Virtual Environment

1. Within your project's virtual environment, download and run `get-pip.py` to ensure you have pip installed. You can find `get-pip.py` [here](https://bootstrap.pypa.io/get-pip.py).

    ```bash
    python get-pip.py
    ```

## Step 4: Install Required Libraries

1. Run the following command in the command line to install all the required libraries for the project:

    ```bash
    pip install -r requirements.txt
    ```

    Note: The guide mentions `requirements.exe`, but typically, the file for pip dependencies is `requirements.txt`. Ensure you have the correct file.

## Step 5: Running the Image Categorization Program

1. `run.py` is the program used to input images for AI categorization. To run it, use the command line with the image path as an argument:

    ```bash
    python run.py C:\Users\Jude\Downloads\mallard.png
    ```

    Replace `C:\Users\Jude\Downloads\mallard.png` with the path to your image.

## Step 6: Training the AI Model

1. `tweakAI.py` is the training program for the AI used in this project. To run the training script, use the command:

    ```bash
    python tweakAI.py
    ```

## Step 7: Exploring Additional Examples and Data Analysis

1. For more examples of training implementation and some data analysis, check out `main.py`. This script contains additional functionalities and examples related to the project.
