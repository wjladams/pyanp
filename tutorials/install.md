# Installing and running Jupyter Notebooks
This tutorial will lead you through:

1. [What is Python, Jupyter, a Jupyter notebook, and Anaconda?](#1-what-is-python-jupyter-and-a-jupyter-notebook)
2. [Installing Anaconda, which includes Python and Jupyter](#2-installing-anaconda)
3. [Running the Jupyter notebook server on Windows, Mac, or Linux systems](#3-running-jupyter)
4. [Creating a new Jupyter notebook, and opening existing Jupyter notebooks.](#4-creating-a-jupyter-notebook)
5. [Installing the `pyanp` library and doing a quick sample calculation to verify `pyanp` is installed correclty.](#5-installing-pyanp)

## 1. What is Python, Jupyter and a Jupyter notebook?

* **Python** is a scripting programming language that is used for many mathematical and computer science research projects, in addition to many other applications.  For more information see https://www.python.org/
* **Jupyter** is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.  For more information see http://jupyter.org/
* **A Jupyter Notebook** is a document in the Jupyter application that includes live code, text (with markdown support), equations (including LaTeX support).  It supports Julia, Python, and R languages behind the scenes (which is where the name came from), with the ability to support many many others.
* **Anaconda** is a distribution of python with loads of extra libraries already bundles (e.g. scipy, pandas, numpy, and many more) that also includes Jupyter.  It is the fastest way to get started with Jupyter and Python.

## 2. Installing Anaconda
The easiest way to install Jupyter and python altogether is to install anaconda:

1. Download the installer from https://www.anaconda.com/download.
1. Choose the **Python 3.6 version**
1. Download and install as you normally would for your operating system.
1. See https://docs.anaconda.com/anaconda/install/ for detailed installation instructions for each operating system (Windows, Mac, and Linux).

## 3. Running Jupyter

See [How to run Jupyter](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html) for instructions.

## 4. Creating a Jupyter Notebook

After you have Jupyter running simply click on the NEW menu button in the upper right corner.  See [this excellent 
getting started medium post for more information](https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46#bc16)

## 5. Installing `pyanp`

Once you have Jupyter and python installed, in a cmd window on Windows (or a terminal on Mac/Linux) run the following command

```
pip install pyanp
```
that will download the latest version of pyanp and install it automatically.
