import configparser
import splitfolders
import shutil
import os
import yaml
from zipfile import ZipFile
from tqdm import tqdm
from glob import glob
from pathlib import Path

def mkdir():
    if not os.path.exists('./split-folders/input/images'):
        os.makedirs('./split-folders/input/images')
            
    if not os.path.exists('./split-folders/input/labels'):
        os.makedirs('./split-folders/input/labels')
        
    if not os.path.exists('./split-folders/output'):
        os.makedirs('./split-folders/output')

def split_folders(config):
    images = glob(f'{config["target"]}/*jpg')
    
    # classes.txt 빼기
    labels = list(map(Path, glob('./data/img/*txt')))
    labels = list(filter(lambda item: item.name != 'classes.txt', labels))

    for src in tqdm(images, ncols = 80, desc="images move"):
        shutil.move(src, dst=f"./split-folders/input/images")
        
    for src in tqdm(labels, ncols = 80, desc="labels move"):
        shutil.move(str(src), dst=f"./split-folders/input/labels")

    ratio = tuple(map(float, config["ratio"].split(',')))
    print(f"ratio: {ratio}")
    splitfolders.ratio(input="./split-folders/input", output="./split-folders/output", seed=1337, ratio=ratio)

def mkyaml():
    classes = []
    
    with open('./classes.txt', 'r') as f:
        line = f.readline()
        while line:
            classes.append(line.strip())
            line = f.readline()
    
    yaml_data = {
        'names': classes,
        'nc': len(classes),
        'test': './test/images',
        'train': './train/images',
        'valid': './valid/images'
    }
    
    with open('./split-folders/output/data.yaml', 'w') as f:
        yaml.safe_dump(yaml_data, f)

def zip_folders():
    print("zipping...")
    with ZipFile('./output.zip', 'w') as zipf:
        for root, _, files in os.walk('./split-folders/output'):
            for file in files:
                file_path = os.path.join(root, file)
                # 상대 경로로 압축 파일에 추가
                zipf.write(file_path, os.path.relpath(file_path, './split-folders/output'))

def delete_folders(folder_path):
    for root, _, files in os.walk(folder_path, topdown=False):
        # 파일 삭제
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

        # 폴더 삭제
        for folder in os.listdir(root):
            folder_path = os.path.join(root, folder)
            if os.path.isdir(folder_path):
                os.rmdir(folder_path)
           
def main(config):
    images = glob(f'{config["target"]}/*jpg')
    
    if not images:
        print("No images.")
        return
    
    mkdir()
    split_folders(config)
    mkyaml()
    zip_folders()
    delete_folders('./split-folders')
    
def argparser():
    config = configparser.ConfigParser()
    config.read('./config.ini')
    return config['split-folders']
    
if __name__ == "__main__":
    main(argparser())