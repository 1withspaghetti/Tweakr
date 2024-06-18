# Tweakr

An app that uses AI to classify fish images as either "locked in" or "tweaking"

For the submodules below, refer to the README in each individual folder for instructions to develop and run each portion.

## Categorization App

This Java Swing app is used to categorize a folder or any fish photos and emulates a grindr like panel to sort each image into the respective bins, locked in or tweaking.
The sorted images can be found in the `output` folder.

## Tweakr AI

The brains of the project, uses python and varies other libraries to utilize the full potential of today's computational power. This app contains scripts to both train the model and predict the classification of images. These scripts are called by the Tweakr Client.

## Tweakr Client

This is that users will see, it's a UI client in Java Swing that allows people to select an image and then call the Tweakr AI to give a classification.