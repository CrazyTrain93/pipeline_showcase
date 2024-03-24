# Gradient Descent

The objective of ANY Machine Learning Model is to find parameters, weights or a structure that minimizes the Cost function.

A way to minimize the Cost function is Gradient Descent. We want to dive deep into this very basic ML method and try to understand Gradient Descent on a very granular level.

## Task

Please work through the notebooks in this particular order:

1. [Gradient_Descent](1_Gradient_Descent.ipynb)
2. [Gradient_Descent_Visuallization](2_Gradient_Descent_Visualization.ipynb)
3. [Gradient_Descent_Codealong](3_Gradient_Descent_Codealong.ipynb)
4. [Bonus_Classification](4_Bonus_Classification.ipynb)

In the first notebook, you will see the code that performs each step of the gradient descent. Try to understand what happens in each line and document each step. The second notebook shows you what gradient descent really looks like (visually), and in the third notebook it's your job to write the code for gradient descent from scratch. In the last notebook, you will see a simple classification example that will give you a little preview of what to expect in the next few days.


## Set up your Environment

Please make sure you have forked the repo and set up a new virtual environment. For this purpose you can use the following commands:

The added [requirements file](requirements.txt) contains all libraries and dependencies we need to execute the Gradient Descent notebooks.

*Note: If there are errors during environment setup, try removing the versions from the failing packages in the requirements file. M1 shizzle.*

### **`macOS`** type the following commands : 


- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-Bash` CLI :
    ```
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```