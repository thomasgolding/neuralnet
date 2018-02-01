from setuptools import setup

setup(name='neuralnet',
      version='0.1',
      description='Simple neural net',
      url='',
      author='Thomas Golding',
      author_email='thomas.golding@gmail.com',
      license='MIT',
      packages=['neuralnet'],
      install_requires=['numpy'],
      test_suite='nose.collector',
      test_require=['nose'],
      zip_safe=False)
