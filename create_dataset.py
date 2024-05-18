import os
import shutil
from tqdm import tqdm
import csv
import random
import pandas as pd

import os

import os

def rename_images(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        # Check for 'benign' or 'malignant' and magnification level in directory path
        if 'benign' in dirpath and any(mag in dirpath for mag in ['40x', '100x', '200x', '400x']):
            label = 'benign'
        elif 'malignant' in dirpath and any(mag in dirpath for mag in ['40x', '100x', '200x', '400x']):
            label = 'malignant'
        else:
            continue
        
        magnifications = ['40x', '100x', '200x', '400x']
        for magnification in magnifications:
            if magnification in dirpath:
                magnification_level = magnification
                break
        
        # Initialize image number for the current magnification and label
        image_number = 1
        for filename in filenames:
            if filename.endswith('.png'):
                source_path = os.path.join(dirpath, filename)
                
                # Determine label from filename
                if 'M_' in filename:
                    label = 'malignant'
                else:
                    label = 'benign'

                new_filename = f'{label}_{magnification_level}_{image_number}.png'
                destination_path = os.path.join(dirpath, new_filename)
                os.rename(source_path, destination_path)
                
                image_number += 1



                
def find_total_BreakHis():
    file_path = '../BreakHis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/'
    count = 0
    for dirs, _, file in os.walk(file_path):
        for f in file:
            k = os.path.join(dirs, f)
            if k.endswith('.png'):
                count += 1
    count = (count / 100) * 5.34
    return count
                
def create_dir():
    os.makedirs('dataset', exist_ok=True)
    
    #Test
    os.makedirs('dataset/test/benign/40x', exist_ok=True)
    os.makedirs('dataset/test/malignant/40x', exist_ok=True)
    
    os.makedirs('dataset/test/benign/100x', exist_ok=True)
    os.makedirs('dataset/test/malignant/100x', exist_ok=True)
    
    os.makedirs('dataset/test/benign/200x', exist_ok=True)
    os.makedirs('dataset/test/malignant/200x', exist_ok=True)
    
    os.makedirs('dataset/test/benign/400x', exist_ok=True)
    os.makedirs('dataset/test/malignant/400x', exist_ok=True)
    
    # os.makedirs('dataset/test/40x/benign', exist_ok=True)
    # os.makedirs('dataset/test/40x/malignant', exist_ok=True)
    
    # os.makedirs('dataset/test/100x/benign', exist_ok=True)
    # os.makedirs('dataset/test/100x/malignant', exist_ok=True)
    
    # os.makedirs('dataset/test/200x/benign', exist_ok=True)
    # os.makedirs('dataset/test/200x/malignant', exist_ok=True)
    
    # os.makedirs('dataset/test/400x/benign', exist_ok=True)
    # os.makedirs('dataset/test/400x/malignant', exist_ok=True)
    
    #Train
    os.makedirs('dataset/train/40x/benign', exist_ok=True)
    os.makedirs('dataset/train/40x/malignant', exist_ok=True)
    
    os.makedirs('dataset/train/100x/benign', exist_ok=True)
    os.makedirs('dataset/train/100x/malignant', exist_ok=True)
    
    os.makedirs('dataset/train/200x/benign', exist_ok=True)
    os.makedirs('dataset/train/200x/malignant', exist_ok=True)
    
    os.makedirs('dataset/train/400x/benign', exist_ok=True)
    os.makedirs('dataset/train/400x/malignant', exist_ok=True)
    
                    

def getLisOfFiles_dir(tot_iter):
    create_dir()
    file_path = '../BreakHis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/'
    choices = ["train", "test"]
    probabilities = [0.8, 0.2] 
    for dirs, _, filenames in tqdm(os.walk(file_path), total = tot_iter):
        # folder = os.path.basename(dirs)
        for f in filenames:
            result = random.choices(choices, weights=probabilities)[0]
            f = os.path.join(dirs, f)
            if result == 'train':
                if 'benign' in dirs:
                    if '40X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/train/40x/benign')
                    if '100X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/train/100x/benign')
                    if '200X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/train/200x/benign')
                    if '400X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/train/400x/benign')
                        
                elif 'malignant' in dirs:
                    if '40X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/train/40X/malignant')
                    if '100X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/train/100X/malignant')
                    if '200X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/train/200X/malignant')
                    if '400X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/train/400X/malignant')
                    
                    
            elif result == 'test':
                if 'benign' in dirs:
                    if '40X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/test/benign/40x')
                    if '100X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/test/benign/100x')
                    if '200X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/test/benign/200x')
                    if '400X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/test/benign/400x')
                        
                elif 'malignant' in dirs:
                    if '40X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/test/malignant/40x')
                    if '100X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/test/malignant/100x')
                    if '200X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/test/malignant/200x')
                    if '400X' in dirs:
                        if f.endswith('.png'):
                            shutil.copy(f, 'dataset/test/malignant/400x')

                    
tot_iter  = find_total_BreakHis()  
getLisOfFiles_dir(tot_iter)
rename_images('dataset')
