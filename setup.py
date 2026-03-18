from setuptools import setup ,find_packages

requirement_lst = []

def get_requirements():


    try:
      with open('requirements.txt') as f:
        
        for line in f.readlines():
            line.strip()

            if line and line != '-e .':
                requirement_lst.append(line)


    except FileNotFoundError :
       print('requirements.txt file not found')


    return requirement_lst

setup(
    name='churn_prediction',
    version='0.0.1',
    author='ayan',
    author_email='sheikhayanahmad710@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)

