#### Personal Neural Network Library

This library serves as a personal learning resource for understanding neural networks and the algorithms and math behind them. The main purpose of this project is self-education and it doesn't serve any specific application beyond my own learning goals.

#### Features

- Implementation of a feedforward neural network
- Input layer and Dense layer
- Various activation functions
- Various loss functions
- Various optimization algorithms
- Various weight initialization techniques

(check documentation to see the full list of implemented components)

#### Technologies used

- Python 3.10.12
- Numpy 1.23.5
- Pandas 2.12.0 (used in examples to manipulate the input data)

#### Installation

Clone the repository:
```
git clone https://github.com/MortenTsinakov/n2rv.git
```
#### Usage

To run the program in a virtual environment, run those commands from the project root directory:

1. Create a virtual environment: ```python3 -m venv venv```
2. Activate the virtual environment: ```source venv/bin/activate```
3. Install necessary dependencies: ```pip install -r requirements.txt```
4. Use the program, e.g run an example:
  ```console
  (venv) foo@bar:~/n2rv$ cd examples               # navigate to examples directory
  (venv) foo@bar:~/n2rv/examples$ python3 iris.py  # run the iris example
  ```
5. When finished, deactivate the virtual environment: ```deactivate```

For more info about how to use different components, check out the documentation. <br>
To build a model, it's easiest to do it in the main.py file.

#### Documentation

Find the documentation for the project on this page: <br>
https://n2rv.notion.site/N2rv-library-c63d9d4cd9e941d3983b1018a8f42fc2?pvs=4
