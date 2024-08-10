from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='plumai',
    version='0.1.7',
    py_modules=['plumai'],
    packages=find_packages(include=['plumai', 'plumai.*']),
    install_requires=parse_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            'plumai=plumai.plumai:main',
        ],
    },
    author='Tam Nguyen',
    author_email='tamnvhustcc@gmail.com',
    description='A tool for deploying and running AI models to Modal',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://x.com/tamnvvn',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    package_data={
        '': ['templates/*'],
    },
    include_package_data=True,
)