import cv2

def apply_blur(image_path):
    # 이미지 불러오기
    img = cv2.imread(image_path)
    
    # 평균 블러 (Average Blur)
    k_size = (5, 5)  # 커널 크기 (5x5)
    avg_blur = cv2.blur(img, k_size)
    
    # 가우시안 블러 (Gaussian Blur)
    k_size = (5, 5)  # 커널 크기 (5x5)
    sigmaX = 0  # X 방향 표준편차 (0이면 자동으로 계산)
    gaussian_blur = cv2.GaussianBlur(img, k_size, sigmaX)
    
    # 중간값 블러 (Median Blur)
    k_size = 5  # 커널 크기 (5x5)
    median_blur = cv2.medianBlur(img, k_size)
    
    # 양방향 블러 (Bilateral Blur)
    d = 9  # 지름 (커널 크기) (9x9)
    sigmaColor = 75  # 색 공간 필터 시그마 값
    sigmaSpace = 75  # 좌표 공간 필터 시그마 값
    bilateral_blur = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    
    # 결과 이미지 출력
    cv2.imshow('Original Image', img)
    cv2.imshow('Average Blur', avg_blur)
    cv2.imshow('Gaussian Blur', gaussian_blur)
    cv2.imshow('Median Blur', median_blur)
    cv2.imshow('Bilateral Blur', bilateral_blur)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r'C:\Users\User\Desktop\Github\Python(영상처리)\ImageBlurring\test.jpg'  # 이미지 파일 경로를 입력해주세요.
    apply_blur(image_path)
