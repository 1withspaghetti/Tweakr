# Guide to Get This Working on Your Machine

## Downloading PyCharm and Create a Project

1. Download and install [PyCharm](https://www.jetbrains.com/pycharm/download/).
2. Open PyCharm and create a new project.

## Import All Files

1. Place all your project files (`run.py`, `tweakAI.py`, `main.py`, `requirements.txt`, and `tut5-model.pt`) into the project directory.

## Set Up Pip

1. Within your project's virtual environment, download and run `get-pip.py` to ensure you have pip installed. You can find `get-pip.py` [here](https://bootstrap.pypa.io/get-pip.py).

    ```bash
    python get-pip.py
    ```

## Installing Required Libraries

1. Run the following command in the command line to install all the required libraries for the project:

    ```bash
    pip install -r requirements.txt
    ```

    Note: The guide mentions `requirements.exe`, but typically, the file for pip dependencies is `requirements.txt`. Ensure you have the correct file.

## Running the Image Categorization Program

1. `run.py` is the program used to input images for AI categorization. To run it, use the command line with the image path as an argument:

    ```bash
    python run.py [your file path]
    ```
    Replace `[your file path]` with the path to your image.

   Alternatively, you can make a custom run config within Pycharm via the edit configurations button in the dropdown under `current file` at the top of the screen. To do this, you can press the `Add new run configuration...` button and select python. Next to script, you input the path to `run.py` and your image file path in the script parameters field. 

   This is good for repeat testing, due to being able to run with one button press (green arrow next to what should now say `run w/ args`)

## Training the AI Model

1. `tweakAI.py` is the training program for the AI used in this project. To run the training script, use the command:

    ```bash
    python tweakAI.py
    ```
   **THIS PROGRAM WILL NOT RUN WITHOUT THE DATA FROM`[folder that tweakAI.py is in]\data\tweakr`**
    
## Additional Examples and Data Analysis

1. For more examples of training implementation and some data analysis, check out `main.py`. This script contains additional functionalities and examples related to the project. 
   
   
   **THIS PROGRAM WILL NOT RUN WITHOUT THE DATA FROM`[folder that main.py is in]\data\CUB_200_2011`**