from setuptools import setup

setup(name='pythonml',
      version='0.1',
      description='ML-algorithms',
      url='',
      author='Thomas Golding',
      author_email='thomas.golding@gmail.com',
      license='MIT',
      packages=['pythonml'],
      install_requires=['numpy', 'scipy'],
      setup_requires=['pytest-runner'],
      test_require=['pytest'])
