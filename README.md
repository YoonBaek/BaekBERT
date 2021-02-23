# BaekBERT
이 모델은 KcBERT Large pretrain 모델을 활용한  
DownStream Task 모델입니다.  
영화 관련 글과 비영화 관련 글을 걸러줍니다.  

# Dataset
1. Train/Validset  
Positive : NSMC 데이터셋 (네이버 영화 감성분석 데이터셋)  
Negative : 자체크롤링한 게임관련 게시글 데이터셋  
  
2. Testset for infer  
Private

# How to use
1. train  
cd traintest  
python main.py

2. test(infer)  
cd traintest  
python infer.py

# Reference
[KcBERT: Korean comments BERT](https://github.com/Beomi/KcBERT)

[Transformers by HuggingFace](https://github.com/huggingface/transformers)
