# How to enjoy this project?

Firstly build and run container:

``` shell
docker build -t forest_pred_image .
```
``` shell
docker run -d --name forest_pred_container -p 80:80 forest_pred_image
```
Then choose right prediction type:
```
http://127.0.0.1:80/simple-heuristic-prediction

http://127.0.0.1:80/nearest-neighbors-prediction

http://127.0.0.1:80/random-forest-prediction

http://127.0.0.1:80/neural-network-prediction
```
...and post JSON file with forest features\
(look at 'data/sampleFeatures.json' format)

### WARNING!
To use random forest or kNN classifiers regenerate models with my scripts.\
Models in this repo were train with other parameters, because files on GitHub must be smaller (less than 100MB)