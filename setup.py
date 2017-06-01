from setuptools import setup

setup(name='cobea',
      version='0.13',
      description='Closed-Orbit Bilinear-Exponential Analysis, an algorithm to be used for studying betatron oscillations in particle accelerators',
      url='http://bitbucket.org/b-riemann/cobea/',
      author='Bernard Riemann',
      author_email='bernard.riemann@tu-dortmund.de',
      license='private',
      packages=['cobea'],
      requires=['numpy', 'scipy'],
      zip_safe=False)
