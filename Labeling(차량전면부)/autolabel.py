import configparser
import os
import uuid
from datetime import datetime
from glob import glob
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

def rename(config):
    img_list = glob(f"{config['target']}/*.jpg")
    
    for file in tqdm(img_list, ncols = 80, desc="rename"):
        time_stamp = datetime.now().strftime("%Y%m%d")
        rand_str = str(uuid.uuid4().hex[:6])
        os.rename(file, f"{config['target']}/{time_stamp}_{rand_str}.jpg")
    
    
def main(config):
    rename(config)
    
    model = YOLO(config["weights"])
    img_list = glob(f"{config['target']}/*.jpg")
    
    for img in tqdm(img_list, ncols = 80, desc="labeling"):       
        results = model.predict(source=img, verbose=False)
        for r in results:
            table = []
            boxes = r.boxes
            for box in boxes:
                row = []
                row.append(int(box.cls))
                for xywh in box.xywhn.squeeze().tolist():
                    row.append(xywh)
                table.append(row)
                
            file = Path(r.path).with_suffix('.txt')
            output = Path(config["output"]) / file.name
            with open(output, 'w') as f:
                for row in table:
                    row_data = ' '.join(map(str, row))
                    f.write(row_data + '\n')
    
    classes = '\n'.join(model.names.values())        
    with open('classes.txt', 'w') as f:
        f.write(classes)
    with open(f"{config['target']}/classes.txt", 'w') as f:
        f.write(classes)
        
def argparser():
    try:
        config = configparser.ConfigParser()
        config.read('./config.ini')
        return config['autolabel']
    
    except:
        print("config.ini를 찾을 수 없습니다.")

if __name__ == "__main__":
    main(argparser())