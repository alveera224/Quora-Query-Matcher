import os

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "data",
        "app/static",
        "app/model/saved_models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    print("Setting up project directories...")
    create_directories()
    print("Directory setup complete!") 