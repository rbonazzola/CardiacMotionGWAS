from setuptools import setup, find_packages

setup(
  name="cardiac-motion-upe",
  version="0.0.1",
  author="Rodrigo Bonazzola (rbonazzola)",
  author_email="rodbonazzola@gmail.com",
  description="Python package for analyzing results of GWAS of cardiac motion unsupervised biomarkers",
  long_description=open("README.md", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  url="https://github.com/rbonazzola/CardiacMotion",
  packages=find_packages(),
  install_requires=[
    "cardiac_motion>=0.0.1",
    "cardio_mesh>=0.0.1"
  ],
  python_requires=">=3.8,<3.11"
)

