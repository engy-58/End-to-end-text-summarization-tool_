import setuptools

with open("README.md"   , "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = 'End-to-end-text-summarization-tool_'
AUTHOR_USER_NAME = 'engy-58'
SRC_REPO = 'text_summarizer'

setuptools.setup(
    name=REPO_NAME + SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    description='experimental tool for text summarization using LLMs.',
    long_description=long_description, 
    long_description_content="text/markdown",
    url="https://github.com/engy-58/End-to-end-text-summarization-tool_",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ]
)