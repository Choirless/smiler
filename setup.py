import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="choirless_smiler",
    version="1.2.1",
    author="Matt Hamilton",
    author_email="mh@quernus.co.uk",
    description="A library and command to extract the smiliest image from a video",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/choirless/smiler",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "tensorflow",
        "dlib",
        "imutils",
        "opencv-python",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "smiler = choirless_smiler.smiler:main",
        ]
    },
)
