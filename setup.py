from distutils.core import setup

setup(
    name='MachineLearning',
    version='0.1.0',
    author='W. Locke',
    author_email='locke+public@getyog.com',
    packages=['machinelearning', 'machinelearning.test'],
    scripts=[],
    url='http://pypi.python.org/pypi/MachineLearning/',
    license='LICENSE.txt',
    description='Useful machine learning stuff.',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.5.1",
    ],
)