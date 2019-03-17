https://st1990.hatenablog.com/entry/2019/03/16/231554?_ga=2.52856360.1866587403.1552700729-1195242149.1551491401

## Deep-Matrix-Factorization-in-Keras
### Overview
-Implement Matrix Factorization with keras and numpy
-Implement Deep Matrix Factorization with keras
-Verify the accuracy of Deep Matrix Factorization with MovieLens

### Accuracy verification result
Prediction method | RMSE (training data) | RMSE (test data)
Random | 1.837 | 1.829
Overall average | 1.064 | 1.055
Movie Unit Average | 0.962 | 0.952
User unit average | 0.951 | 0.953
Matrix Factorization | 0.815 | 0.816
Deep Matrix Factorization (1) | 0.815 | 0.815
Deep Matrix Factorization (2) | 0.814 | 0.815
Deep Matrix Factorization (3) | 0.839 | 0.836
Deep Matrix Factorization (4) | 0.795 | 0.798

<Notes>
(1) Structure in which the interaction input part is Deep
(2) The structure where the part after the interaction is Deep
(3) Structure that made (2) like ResNet
(4) Structure using both (1) and (2)


## Deep-Matrix-Factorization-in-Keras
### 概要
- kerasとnumpyでMatrix Factorizationを実装
- kerasでDeep Matrix Factorizationを実装
- MovieLensでDeep Matrix Factorizationの精度を検証

### 精度検証結果
予測方法                     |RMSE (training data)|RMSE (test data)
Random                      | 1.837| 1.829
全体平均                     | 1.064| 1.055
映画単位平均                 | 0.962| 0.952
ユーザ単位平均               | 0.951| 0.953
Matrix Factorization        | 0.815| 0.816
Deep Matrix Factorization(1)| 0.815| 0.815
Deep Matrix Factorization(2)| 0.814| 0.815
Deep Matrix Factorization(3)| 0.839| 0.836
Deep Matrix Factorization(4)| 0.795| 0.798

<注釈>
(1) 相互作用の入力部分をDeepにした構造
(2) 相互作用以降の部分をDeepにした構造
(3) (2)をResNetのようにした構造
(4) (1)と(2)の両方を使った構造

