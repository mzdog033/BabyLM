from setuptools import setup, find_packages

setup(
  name = 'pt_framework',
  package_dir={"": "src"},
  packages=find_packages("src"),
  version = '1.0.0',
  license='MIT',
  description = 'Pytorch Training Framework',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'deep learning',
  ],
  install_requires=[
    'pymongo',
    'GitPython',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
