from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="conversational-drift-analyzer",
    version="1.0.0",
    author="NLP Engineer",
    author_email="developer@example.com",
    description="An end-to-end NLP framework to measure, detect, and visualize conversational drift in multi-turn dialogues",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/conversational-drift-analyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Communications :: Chat",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "pandas>=1.5.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "transformers": ["sentence-transformers>=2.2.0"],
        "ui": ["streamlit>=1.25.0", "plotly>=5.15.0"],
        "api": ["fastapi>=0.100.0", "uvicorn>=0.23.0"],
        "all": [
            "sentence-transformers>=2.2.0",
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "pytest>=7.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "drift-analyzer=conversational_drift.cli:main",
        ],
    },
)
