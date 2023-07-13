import os

folder_path = r'C:\Users\User\Desktop\Github\YOLO(딥러닝)\YOLOv8\real-1\train\labels'  # 폴더 경로 설정
new_name_prefix = 'train_'  # 새로운 이름의 접두사

file_list = os.listdir(folder_path)  # 폴더 내의 파일 목록 가져오기

for index, file_name in enumerate(file_list):
    if file_name.endswith('.txt') or file_name.endswith('.png'):  # 이미지 파일인 경우만 이름 변경
        file_path = os.path.join(folder_path, file_name)
        new_file_name = f'{new_name_prefix}{index}.txt'  # 새로운 파일 이름 생성
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(file_path, new_file_path)  # 파일 이름 변경

print('이미지 파일 이름 변경이 완료되었습니다.')