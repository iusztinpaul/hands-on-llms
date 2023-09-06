def read_requirements(file_path):
    with open(file_path, "r") as file:
        requirements = [line.strip() for line in file if line.strip()]

    return requirements
