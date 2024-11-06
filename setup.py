from setuptools import setup, find_packages

setup(
    name='carbonia',
    version='0.1.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'your_command=match.match:main_function',  # Replace with your main function
        ],
    },
    author='Hugo Cruz',
    author_email='huggcruzz@gmail.com',
    description='Using AI to make a match between an input dataframe and a target dataframe. Useful for data matching, specially when the target data is a carbon database (ecoinvent, L1P5 etc)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/hugocruzz/carbonia',  # Replace with your GitHub URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
    'pandas',
    'numpy',
    'scikit-learn',
    'openai',
    'tiktoken',
    'tableauhyperapi',
    'python-dotenv',
    'pyyaml',
    'logging'
    ],
    python_requires='>=3.6',
)