#파일 이름에 있는 한글을 영어로 변경
import os
import re

def rename_korean_to_english(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # .jpg 파일만 대상으로 합니다. 다른 형식도 포함하려면 조건을 추가해야 합니다.
            # 한글 부분을 영어로 바꾸는 정규식을 사용합니다.
            new_filename = re.sub(r'[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+', 'english', filename)
            src = os.path.join(folder_path, filename)  # 원본 파일의 경로
            dst = os.path.join(folder_path, new_filename)  # 새로운 파일의 경로
            os.rename(src, dst)  # 파일 이름 변경

if __name__ == '__main__':
    folder_path = r"C:\Users\User\Desktop\Github\YOLO(딥러닝)\YOLOv8\Anaconda\testfile"  # 이미지 파일들이 있는 폴더의 경로로 바꿔주세요
    rename_korean_to_english(folder_path)