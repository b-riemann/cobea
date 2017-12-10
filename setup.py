from setuptools import setup

setup(name='cobea',
      version='0.25',
      description='Closed-Orbit Bilinear-Exponential Analysis, an algorithm for studying betatron oscillations in particle accelerators',
      url='https://github.com/b-riemann/cobea',
      author='Bernard Riemann',
      author_email='bernard.riemann@tu-dortmund.de',
      license='Simplified BSD License',
      packages=['cobea'],
      requires=['scipy'],
      zip_safe=False)
