# Standard Python packaging modules - available in all Python installations
try:
    from setuptools import setup, find_packages  # type: ignore
except ImportError:
    from distutils.core import setup  # type: ignore
    def find_packages():
        return ['merit_analyzer']

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="merit-analyzer",
    version="1.0.0",
    author="Merit Analyzer Team",
    author_email="team@merit-analyzer.com",
    description="AI system test failure analysis and recommendation engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/merit-analyzer/merit-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "merit-analyze=merit_analyzer.cli:main",
        ],
    },
    include_package_data=True,
)
