## MRR 성능 이슈 및 원인 분석

### 기존 main.py 실행 순서
1. seed 고정  
2. data_loader 생성  
3. model 정의  

→ 이 경우, Jupyter 환경에서 독립 루프로 테스트했을 때보다 MRR 성능이 낮게 나옴

---

### Jupyter 환경에서의 실행 방식
- `seed 고정` 직후에 바로 `model 정의`
- 이후 독립 루프로 테스트 수행

---

### 원인 추정
- model head에 **learnable query**가 존재
- 이 query 초기화가 **RNG 상태(seed 소비 순서)**에 크게 의존함
- main.py에서는 `data_loader`를 먼저 생성하면서  
  RNG가 이미 소모된 상태에서 model이 초기화됨
- 그 결과, learnable query의 초기값이  
  Jupyter 독립 루프에서의 값과 크게 달라짐

---

### 해결 방법
main.py 실행 순서를 아래와 같이 변경

1. seed 고정  
2. model 정의  
3. data_loader 생성  

→ learnable query 초기값이 Jupyter 환경과 유사해짐  
→ MRR 성능 개선 확인