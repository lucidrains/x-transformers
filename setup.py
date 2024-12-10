from setuptools import setup, find_packages

setup(
  name = 'x-transformers',
  packages = find_packages(exclude=['examples']),
  version = '1.42.26',
  license='MIT',
  description = 'X-Transformers - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/x-transformers',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  install_requires=[
    'einx>=0.3.0',
    'einops>=0.8.0',
    'loguru',
    'packaging>=21.0',
    'torch>=2.0',
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
