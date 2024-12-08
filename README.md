# verifier
LLM reasoner


# 주요 이슈 기록
SFT Trainer로 학습하는 constant length로 학습되고, 이 데이터들은 주로 padding이 없다. 그래서 batch generation 시에 padding이 있으면 문제가 발생한다.
그래서 방법은 1) 학습시에도 padding을 넣어주던지, 2) 1개씩만 생성하는 것이다.