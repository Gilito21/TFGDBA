import os

def find_obj_files(start_directory='/home/ubuntu'):
    """Utility to find OBJ files on your server"""
    print(f"Searching for OBJ files starting from: {start_directory}")
    
    results = []
    for root, dirs, files in os.walk(start_directory):
        # Skip hidden directories and virtual environments to speed up search
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv' and d != 'env']
        
        for file in files:
            if file.endswith('.obj'):
                full_path = os.path.join(root, file)
                results.append(full_path)
    
    return results

# Run the search
obj_files = find_obj_files()

# Print results
print(f"Found {len(obj_files)} OBJ files:")
for file_path in obj_files:
    print(f"- {file_path}")

# Specifically look for Porsche models
porsche_files = [f for f in obj_files if 'porsche' in f.lower()]
print(f"\nFound {len(porsche_files)} Porsche OBJ files:")
for file_path in porsche_files:
    print(f"- {file_path}")
