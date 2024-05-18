import shutil
import os
from tqdm import tqdm


def copy_test_images_to_train(source_root, target_root):
    magnifications = ['40x', '100x', '200x', '400x']
    categories = ['benign', 'malignant']
    
    for magnification in magnifications:
        for category in categories:
            source_dir = os.path.join(source_root, 'test', category, magnification)
            target_dir = os.path.join(target_root, magnification, category)
            
            # Create target directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            # Check if source directory exists
            if os.path.exists(source_dir):
                # Copy files from source to target directory
                for file in os.listdir(source_dir):
                    source_file = os.path.join(source_dir, file)
                    target_file = os.path.join(target_dir, file)
                    shutil.copy(source_file, target_file)
                print(f"Files copied from {source_dir} to {target_dir}")
            else:
                print(f"Source directory {source_dir} does not exist.")


def reorganize_query_images(base_dir):
    magnifications = ['40x', '100x', '200x', '400x']
    categories = ['benign', 'malignant']
    
    # Create new directory structure
    for magnification in magnifications:
        for category in categories:
            new_dir = os.path.join(base_dir, magnification, category)
            os.makedirs(new_dir, exist_ok=True)
    
    # Move files to the new directory structure
    for category in categories:
        for magnification in magnifications:
            old_dir = os.path.join(base_dir, category, magnification)
            new_dir = os.path.join(base_dir, magnification, category)
            if os.path.exists(old_dir):
                for file in os.listdir(old_dir):
                    shutil.move(os.path.join(old_dir, file), new_dir)
                    
    # Clean up empty old directories
    for category in categories:
        for magnification in magnifications:
            old_dir = os.path.join(base_dir, category, magnification)
            if os.path.exists(old_dir) and len(os.listdir(old_dir)) == 0:
                os.rmdir(old_dir)

def create_and_populate_database(source_root, target_root, magnifications, categories):
    for magnification in tqdm(magnifications, desc="Processing magnifications"):
        for category in tqdm(categories, desc=f"Processing categories for {magnification}"):
            # Create target directory structure
            target_dir = os.path.join(target_root, magnification, category)
            os.makedirs(target_dir, exist_ok=True)
            
            # Define source directory
            source_dir_train = os.path.join(source_root, 'train', magnification, category)
            source_dir_test = os.path.join(source_root, 'test', category, magnification)
            
            # Copy files from train and test to the database directory
            for source_dir in [source_dir_train, source_dir_test]:
                if os.path.exists(source_dir):
                    for file in tqdm(os.listdir(source_dir), desc=f"Copying files from {source_dir}"):
                        shutil.copy(os.path.join(source_dir, file), target_dir)

# Define source and target roots
# source_root = 'dataset'
# target_root = 'database'
# magnifications = ['40x', '100x', '200x', '400x']
# categories = ['benign', 'malignant']

# # Populate the database directory with images from dataset train and test
# create_and_populate_database(source_root, target_root, magnifications, categories)
# base_query_dir = 'query'
# reorganize_query_images(base_query_dir)
# source_root = 'dataset/'
# target_root = 'test_set/'
# copy_test_images_to_train(source_root, target_root)


