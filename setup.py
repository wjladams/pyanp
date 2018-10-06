'''
Created October 24, 2017

@author: Bill Adams
'''

from setuptools import setup

setup(name='pyanp',
      version='0.3.1',
      description='Math library for ANP/AHP calculations',
      url='https://github.com/wjladams/pyanp',
      author='Bill Adams',
      author_email='wjadams@decisionlens.com',
      license='MIT',
      package_dir={'pyanp':'pyanp'},
      python_requires=">=3.6",
      install_requires=['numpy', 'pandas'],
      packages=['pyanp'],
      #py_modules=['fxmodel'],
      #package_data={'bamath': ['data/*.csv', 'data/*.pickle', 'data/*.json', '*.css']},
      zip_safe=False)
