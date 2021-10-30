from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="breaker_audio",
    version="0.0.3",
    packages=find_packages(),
    package_data={},
    python_requires=">=3.5",
    install_requires=requirements,
    author="Jaap Oosterbroek",
    author_email="jaap.oosterbroek@gmail.com",
    description="lots of fun audio class mostly related to voice manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kozzion/breaker_audio",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)