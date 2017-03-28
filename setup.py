from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='helium_stark',
      version='0.0.1',
      description='Calculate the Stark effect in Rydberg helium using the Numerov method',
      url='',
      author='Adam Deller',
      author_email='a.deller@ucl.ac.uk',
      license='BSD 3-clause',
      packages=['helium_stark'],
      include_package_data=True,
      zip_safe=False)
