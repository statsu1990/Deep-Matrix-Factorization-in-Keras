https://st1990.hatenablog.com/entry/2019/03/16/231554?_ga=2.52856360.1866587403.1552700729-1195242149.1551491401<br>

## Deep-Matrix-Factorization-in-Keras<br>
### Overview<br>
- Implement Matrix Factorization with keras and numpy<br>
- Implement Deep Matrix Factorization with keras<br>
- Verify the accuracy of Deep Matrix Factorization with MovieLens<br>

### Accuracy verification result<br>
Prediction method | RMSE (training data) | RMSE (test data)<br>
Random | 1.837 | 1.829<br>
Overall average | 1.064 | 1.055<br>
Movie Unit Average | 0.962 | 0.952<br>
User unit average | 0.951 | 0.953<br>
Matrix Factorization | 0.815 | 0.816<br>
Deep Matrix Factorization (1) | 0.815 | 0.815<br>
Deep Matrix Factorization (2) | 0.814 | 0.815<br>
Deep Matrix Factorization (3) | 0.839 | 0.836<br>
Deep Matrix Factorization (4) | 0.795 | 0.798<br>

<Notes><br>
(1) Structure in which the interaction input part is Deep<br>
(2) The structure where the part after the interaction is Deep<br>
(3) Structure that made (2) like ResNet<br>
(4) Structure using both (1) and (2)<br>


## Deep-Matrix-Factorization-in-Keras<br>
### 概要<br>
- kerasとnumpyでMatrix Factorizationを実装<br>
- kerasでDeep Matrix Factorizationを実装<br>
- MovieLensでDeep Matrix Factorizationの精度を検証<br>

### 精度検証結果<br>
予測方法                     |RMSE (training data)|RMSE (test data)<br>
Random                      | 1.837| 1.829<br>
全体平均                     | 1.064| 1.055<br>
映画単位平均                 | 0.962| 0.952<br>
ユーザ単位平均               | 0.951| 0.953<br>
Matrix Factorization        | 0.815| 0.816<br>
Deep Matrix Factorization(1)| 0.815| 0.815<br>
Deep Matrix Factorization(2)| 0.814| 0.815<br>
Deep Matrix Factorization(3)| 0.839| 0.836<br>
Deep Matrix Factorization(4)| 0.795| 0.798<br>

<注釈><br>
(1) 相互作用の入力部分をDeepにした構造<br>
(2) 相互作用以降の部分をDeepにした構造<br>
(3) (2)をResNetのようにした構造<br>
(4) (1)と(2)の両方を使った構造<br>

