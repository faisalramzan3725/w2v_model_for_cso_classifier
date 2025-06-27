from setuptools import setup, find_packages

setup(
    name="kmi_cso_upgradation",
    version="0.1.0",
    description="This project leverages the Computer Science Ontology (CSO) to process academic paper metadata, identify key research topics, and generate word embeddings using Word2Vec for semantic analysis and downstream NLP tasks.",
    author="Faisal Ramzan",
    author_email="faisal.ramzan@unica.it",
    url="https://github.com/faisalramzan3725/kmi_cso_upgradation",
    packages=find_packages(),
    install_requires=[
        "gensim==4.3.3",
        "nltk==3.9.1",
        "simplejson==3.20.1",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "scipy==1.13.1",
        "matplotlib==3.10.3",
        "jupyter==1.0.0",
        "ipython==9.2.0",
    ],
    python_requires=">=3.11, <3.12",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)