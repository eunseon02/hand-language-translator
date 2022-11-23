# hand-language-translator
미디어파이프(mediapipe)를 이용한 수화 해석 알고리즘

##1) dataset 수집

![hand_landmarks](https://user-images.githubusercontent.com/108911413/203527185-404056b5-4bad-4139-ab6a-bc1f4aa8d316.png)

데이터의 수집은 다음과 같이 이루어진다.
	- 손 landmark의 3차원 좌표: 21개 x3
	- 각 동작을 구분하기 위한 라벨: 1개
  
  구어체와 문어체에서 사용 빈도가 높은 한글의 자소 19가지를 선별하여 자음 10개 및 모음 9개를 택하여 학습시켰다.
  
