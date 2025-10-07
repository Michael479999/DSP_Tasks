# Discrete Signal Tool Application

## Introduction

This is a Discrete Signal Tool application for visualizing and manipulating discrete-time signals. The application is written in Python using the PySide6 library for its graphical user interface (GUI).

## Instructions

To use the application, follow these steps:

#### Loading Signals

1. Load a signal from a text file by clicking the ```Load signal from file``` button.

#### Selecting Signals

2. Select a signal from the list by clicking on the desired signal.

#### Performing Operations

3. Click on a button to perform an operation on the selected signal.

#### Visualizing Results

4. The application will display the result of the operation in the plot area.

#### Saving Results

5. To save the result to a new text file, click on the "Save" button.

#### Folding Signals

6. To fold a signal, select the desired signal and click on the "Fold" button.

#### Visualizing Multiple Signals

7. To visualize multiple signals, select the desired signals and click on the "Plot all" button.

## Signal File Format

The application requires that any signal file loaded must follow the following format:

#### Number of Samples

* The first line of the file must contain an integer N, which is the number of samples in the signal.

#### Sample Data

* The next N lines of the file must contain two integers per line, separated by a space. The first integer is the index of the sample, and the second integer is the value of the sample.

For example, a signal file with 5 samples would have the following format:

5<br>
0 1<br>
1 2<br>
2 4<br>
3 8<br>
4 16<br>