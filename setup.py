from setuptools import setup, find_packages

setup(
    name='detect-shoot',
    version='0.1',
    author='Vedant Mathur',
    author_email='vedant@mathur.ca',
    description='A face detection project using OpenCV',
    packages=find_packages(),
    install_requires=[
        'opencv-python==4.7.0.72',
        'numpy==1.24.2',
    ],
)