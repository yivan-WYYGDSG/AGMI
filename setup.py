import os
import subprocess
from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(
        name='drp',
        version=1.0,
        description='',
        long_description='',
        long_description_content_type='',
        maintainer='',
        maintainer_email='',
        keywords='',
        url='',
        packages=find_packages(exclude=('configs', 'tools')),
        include_package_data=True,
        classifiers=[
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        license='',
        zip_safe=False)
