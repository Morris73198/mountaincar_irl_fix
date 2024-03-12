This may fix some problem we meet in https://github.com/reinforcement-learning-kr/lets-do-irl
This is base on https://github.com/reinforcement-learning-kr/lets-do-irl


在執行app/train.py 時，可能會遇到一些問題，基本上是環境因素所導致，我們提供了train_env.txt，是一個能運行train.py的環境。

然而，同一個環境無法執行test.py，我們提供了可以運行test.py的環境，以及修改後的test.py(test_fix.py)。

要成功的執行let do irl 可先在train_env執行train.py，將輸出的檔案交由test.py執行。
