from setuptools import setup, find_packages

setup(
    name="cancer_project",
    version="0.1.0",
    description="Cancer prediction machine learning project",
    author="karthik kondagari",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "flask",              # remove if not needed
    ],
    python_requires=">=3.7",
)
