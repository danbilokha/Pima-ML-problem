# Diabetes detection

## Training details & Results
1. Best model is ***LogisticRegression*** (in repository is provided).
   - ***accuracy***: 0.7922077922077922, 
   - ***sensivity***: 0.814,
   - ***desicion making threshold***: 0.35. 
2. ***Train / test split***: 0.8 / 0.2

**Further improvents**:
- [ ] try different ansambling approaches;
- [ ] use boosting kind algorithms;
- [ ] extend the dataset with new data in order to create useful in real life software.

## Overview
In the repository is provided:

1. Comprehensive visualization of the dataset in [PimaVisualization.ipynb](PimaVisualization.ipynb).
2. Train and evaluate models for diabetes prediction. Run the next command in order to get train:

```shell script
python3 index.py
```

## Problem overview and What is diabetes ? 
According to NIH, "**Diabetes** is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy. Sometimes your body doesn’t make enough—or any—insulin or doesn’t use insulin well. Glucose then stays in your blood and doesn’t reach your cells.

Over time, having too much glucose in your blood can cause health problems. Although diabetes has no cure, you can take steps to manage your diabetes and stay healthy.

Sometimes people call diabetes “a touch of sugar” or “borderline diabetes.” These terms suggest that someone doesn’t really have diabetes or has a less serious case, but every case of diabetes is serious.

**What are the different types of diabetes?**
The most common types of diabetes are type 1, type 2, and gestational diabetes.

**Type 1 diabetes**
If you have type 1 diabetes, your body does not make insulin. Your immune system attacks and destroys the cells in your pancreas that make insulin. Type 1 diabetes is usually diagnosed in children and young adults, although it can appear at any age. People with type 1 diabetes need to take insulin every day to stay alive.

**Type 2 diabetes**
If you have type 2 diabetes, your body does not make or use insulin well. You can develop type 2 diabetes at any age, even during childhood. However, this type of diabetes occurs most often in middle-aged and older people. Type 2 is the most common type of diabetes.

**Gestational diabetes**
Gestational diabetes develops in some women when they are pregnant. Most of the time, this type of diabetes goes away after the baby is born. However, if you’ve had gestational diabetes, you have a greater chance of developing type 2 diabetes later in life. Sometimes diabetes diagnosed during pregnancy is actually type 2 diabetes.

**Other types of diabetes**
Less common types include monogenic diabetes, which is an inherited form of diabetes, and cystic fibrosis-related diabetes .


## References for learning 

1. https://www.kaggle.com/kernels/scriptcontent/27685252/download
2. https://en.wikipedia.org/wiki/Body_mass_index
3. http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/
4. https://www.news-medical.net/health/What-is-Diabetes.aspx
5. https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
6. http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
7. https://www.scikit-yb.org/en/latest/api/classifier/threshold.html
8. https://www.analyticsindiamag.com/why-is-random-search-better-than-grid-search-for-machine-learning/
9. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
10. https://scikit-learn.org/stable/modules/preprocessing_targets.html#preprocessing-targets
11. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
12. https://twitter.com/bearda24
13. https://www.slideshare.net/DhianaDevaRocha/qcon-rio-machine-learning-for-everyone
14. https://medium.com/@sebastiannorena/some-model-tuning-methods-bfef3e6544f0
15. https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes
16. https://en.wikipedia.org/wiki/Pima_people
