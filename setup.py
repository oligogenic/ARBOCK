from setuptools import setup, find_packages

setup(
    name="ARBOCK",
    version="1.0.1",
    author="Alexandre Renaux",
    description="ARBOCK: A method for mining association rules from a biological KG and leveraging them for predicting pathogenic gene interactions",
    packages=find_packages(),
    install_requires=[
        'networkx>=2.8',
        'scikit-learn>=1.1',
        'seaborn>=0.12',
        'imbalanced-learn>=0.9',
        'scipy>=1.9',
        'pandas>=1.4',
        'pickledb>=0.9',
        'pylibmc>=1.6',
        'tqdm>=4.64',
        'click>=8.1'
    ],
)
