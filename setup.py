from setuptools import setup, find_packages

setup(
    name="ml-evaluator",
    version="1.0.1",
    description="Model Evaluation Toolkit — bias-variance, ROC, model summary, multi-model comparison",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy", "pandas", "matplotlib", "scikit-learn",
    ],
)
