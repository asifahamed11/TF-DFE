## TF-DFE: Topology-Fractal Dynamic Fuzzy Ensemble

### Requirements
pip install -r requirements.txt

### Pipeline (run in order)
1. python 01_somatic_variant_processor.py
2. python 02_remove_missing_values.py
...

### Data Availability
- Final dataset (207,266 variants): https://zenodo.org/doi/10.5281/zenodo.XXXXXXX
- Download external files: [links above]
```

---

## ✅ requirements.txt এ যা থাকবে
```
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
shap
giotto-tda
biopython
pyfaidx
matplotlib
seaborn
scipy
tqdm
joblib
