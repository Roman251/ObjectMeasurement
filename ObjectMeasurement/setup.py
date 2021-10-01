import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setuptools.setup(
    name='ObjectMeasurement',
    version='1.1.1',
    author="Roman Regmi",
    description="object-measuring",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Roman251/ObjectMeasurement",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=['opencv-python>=4.5.1.48', 'imutils>=0.5.4', 'scipy>=1.6.2', 'numpy>=1.21.0'],       
)