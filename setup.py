from setuptools import setup

setup(name='neuralnet',
      version='0.1',
      description='ML-algorithms',
      url='',
      author='Thomas Golding',
      author_email='thomas.golding@gmail.com',
      license='MIT',
      packages=['neuralnet'],
      install_requires=['numpy', 'scipy'],
      setup_requires=['pytest-runner'],
      test_require=['pytest'])
