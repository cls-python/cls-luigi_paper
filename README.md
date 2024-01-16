# cls-luigi-paper


This repository is the official implementation of the examples of the paper:
[CLS-Luigi: A library for automatic analytics pipelines generation](https://i.pinimg.com/474x/fc/27/fb/fc27fb16e1e692e07f8bb3764dfc633b.jpg)

```
@article{meyer2024clsluigi,
  title={CLS-Luigi: A library for automatic analytics pipelines generation},
  author={},
  journal={},
  year={2024}
}
``` 

## Python virtual environments and dependencies

Note that we used two separate virtual environments to execute the examples in this repository. This is because Auto-Sklearn doesn't support python > 3.9 while CLS-Luigi requires Python 3.11.

for running the examples "automl_in_cls_luigi" & "predict_then_optimize" we used Python 3.11.6. You may install the dependencies as follows: 

```
git clone --branch MPC https://github.com/khalil-research/PyEPO.git
pip install PyEPO/pkg/.

pip install -r requirements.txt

```

for running the example in "auto-sklearn" we used Python 3.8.18. You may then install Auto-sklearn as follows:

```
pip install auto-sklearn==0.15.0
```
If you are having problems with downloading Auto-Sklearn, please consult their  [Github-Repository](https://github.com/automl/auto-sklearn)






## Running the AutoML  in CLS-Luigi example
````
cd autosklearn_in_cls_luigi 
python main.py 
python utils/score_collector.py

````

## Running the Predict-Then-Optimize in CLS-Luigi example


````
cd predict_then_optimize
python main.py
python collect_summaries.py
python visualize.py
````


## Running the Auto-Sklearn example
````
cd predict_then_optimize
python main.py
python collect_summaries.py
python visualize.py
````




