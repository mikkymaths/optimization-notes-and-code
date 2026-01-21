# Optimization Notes and Code

This repository contains introductory implementations and notes on
classical optimization algorithms, with a focus on gradient-based
methods and conjugate gradient techniques.

## Contents
- Gradient Descent
- Linear Conjugate Gradient Method
- Nonlinear Conjugate Gradient Methods
- Simple numerical experiments

## Languages and Tools
- Python (NumPy, SciPy)
- MATLAB (for reference and comparison)

## Purpose
This repository is intended as a learning and transition space for
implementing numerical optimization algorithms in Python and preparing
for open-source contributions.
## Example Usage

Basic usage of the conjugate gradient method in Python:

```python
from python.conjugate_gradient import conjugate_gradient
import numpy as np

A = np.array([[4.0, 1.0],
              [1.0, 3.0]])
b = np.array([1.0, 2.0])

x = conjugate_gradient(A, b)
print(x)
