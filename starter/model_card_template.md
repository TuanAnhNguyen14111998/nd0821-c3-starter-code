# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Trained by: Nguyen Tuan anh
- Date: 27/05/2022
- Version: 1
- Algorithm: Random Forest classifier
- Paper: NA
- License: NA

## Intended Use
- Uses: Predicting whether individual earns below or above $50k
- Users: Marketing targeting
- Out of scope: NA

## Training Data
- Categorical Features:
  - workclass
  - education
  - marital-status
  - occupation
  - relationship
  - race
  - sex
  - native-country
- Numerical Features:
  - fnlgt
  - age
  - capital-gain
  - capital-loss
  - hours-per-week
  - education-num
- target
  - salary

## Evaluation Data
- Hold-out of 20%
- Random State: 42

## Metrics
- Precision: 0.74
- Recall: 0.62
- F1 Score: 0.68

## Ethical Considerations
- Based on a sample at a single point in time.

## Caveats and Recommendations
- Small dataset of 32k instances.
