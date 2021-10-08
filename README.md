# ai-hackathon
(광운대x데이콘 AI해커톤 대회)https://dacon.io/competitions/official/235815/overview/description


## Model Architecture
![슬라이드1](https://user-images.githubusercontent.com/46340424/136568847-347196a4-e53e-4e04-9674-4cd187ceb155.PNG)

## Benchmark
|Model|CV|Public|
|-----|--|------|
|LGBM(10-Fold)|0.6784|0.6247|
|Cat(10-fold)|0.5701|0.5777|
|LightAutoML(10-fold)|0.5610|0.5398|
|Stacking-XGB(10-Fold)|0.4808|0.4417|

## Requirement
+ numpy
+ pandas
+ lightgbm
+ xgboost
+ catboost
+ optuna
+ hydra
+ neptune-ai

## Score
Public 1, Private 2
