
from setuptools import setup, find_packages


setup(name='basics',
      version='0.0',
      description='Finds bubbles and holes in HI emission.',
      author='Eric Koch',
      author_email='koch.eric.w@gmail.com',
      url='https://github.com/radio-astro-tools/signal-id',
      scripts=[],
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
       )
