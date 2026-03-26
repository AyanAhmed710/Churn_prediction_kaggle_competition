from setuptools import setup ,find_packages
from typing import List

requirement_lst = []

def get_requirements() -> List[str]:
    try: 
        requirements_lst = []
        with open('requirements.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()  # ← assign it back
                if line and line != '-e .':
                    requirements_lst.append(line)

    except FileNotFoundError:
        print("Requirements file not found.")

    return requirements_lst

setup(
    name='churn_prediction',
    version='0.0.1',
    author='ayan',
    author_email='sheikhayanahmad710@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)

