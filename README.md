# hand-language-translator
미디어파이프(mediapipe)를 이용한 수화 해석 알고리즘


## 1) dataset 수집


데이터의 수집은 다음과 같이 이루어진다.
	- 손 landmark의 3차원 좌표: 21개 x3
	- 각 동작을 구분하기 위한 라벨: 1개
  
  구어체와 문어체에서 사용 빈도가 높은 한글의 자소 19가지를 선별하여 자음 10개 및 모음 9개를 택하여 학습시켰다. 선택된 19가지의 수화동작은 아래와 같다.
  
![ghvhgvkh](https://user-images.githubusercontent.com/108911413/203528090-568f7a56-fa0b-42a3-9446-587870675a38.gif)


## 2) MediaPipe를 이용하여 손가락의 움직임 인식


![hand_landmarks](https://user-images.githubusercontent.com/108911413/203527185-404056b5-4bad-4139-ab6a-bc1f4aa8d316.png)

- 0번 landmark의 좌표를 (0, 0, 0)으로 맞춘다.
- 0번과 5번을 이은 벡터를 미리 정한 기준 벡터의 방향과 크기로 변환해준다.

벡터 변환은 3D 벡터 회전 변환 행렬을 이용한다. 변환하고자 하는 정점에 아래의 행렬을 단순하게 곱하면 연산이 완료된다. 이 과정을 통해 인식되는 손의 크기나 회전에 대하여 데이터가 추가 연산 되며 변환된 3차원 좌표의 값은 넘파이 배열의 형태로 저장한다.

![벡터변환](https://user-images.githubusercontent.com/108911413/203529089-07774ef4-5cdb-4937-96ed-7e2ab203d18a.png)
