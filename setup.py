from setuptools import setup, find_packages # type: ignore
Hypen_e_dot = "-e ."
def get_requirements(file_path: str) -> list[str]:
    """This function will return the list of requirements"""
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if Hypen_e_dot in requirements:
            requirements.remove(Hypen_e_dot)
    return requirements

setup(
    name="ml_project",
    version="0.0.1",
    author="Chethan",
    author_email="chethandl50@gmail.com",
    description="A machine learning project",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)