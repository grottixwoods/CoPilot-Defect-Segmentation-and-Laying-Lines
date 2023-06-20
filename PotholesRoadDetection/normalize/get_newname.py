import os
import random
import string

directory_path = "C:/Users/Grotti/Desktop/DATA/dataset/"

def generate_unique_id(existing_ids):
    unique_id = ''.join(random.choices(string.digits, k=12))
    while unique_id in existing_ids:
        unique_id = ''.join(random.choices(string.digits, k=12))
    return unique_id

def rename_files(directory):
    existing_ids = set()
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):

            unique_id = generate_unique_id(existing_ids)
            existing_ids.add(unique_id)
            new_filename = f"PotholesData2{unique_id}.jpg"
            
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"Renamed '{filename}' to '{new_filename}'")
    
    print("All files have been renamed.")


rename_files(directory_path)