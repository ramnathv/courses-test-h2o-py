---
title: 'Test H2O in Python'
description: 'Chapter description goes here.'
---

## Initialize H2O

```yaml
type: NormalExercise
key: e8c1edbe67
lang: python
xp: 100
skills: 2
```

This is an example exercise.

`@instructions`


`@hint`


`@pre_exercise_code`
```{python}
import h2o
h2o.init()
```

`@sample_code`

```{python}

```

`@solution`

```{python}

```

`@sct`

```{python}

```

---

## GLM Demo

```yaml
type: NormalExercise
lang: python
xp: 100
skills: 2
```

This is an example exercise.

`@instructions`


`@hint`


`@pre_exercise_code`

```{python}
import h2o
h2o.init()
```

`@sample_code`

```{python}
prostate = h2o.load_dataset("prostate")
prostate.describe()
# Randomly split the dataset into ~70/30, training/test sets
train, test = prostate.split_frame(ratios=[0.70])

# Convert the response columns to factors (for binary classification problems)
train["CAPSULE"] = train["CAPSULE"].asfactor()
test["CAPSULE"] = test["CAPSULE"].asfactor()

# Build a (classification) GLM
from h2o.estimators import H2OGeneralizedLinearEstimator
prostate_glm = H2OGeneralizedLinearEstimator(family="binomial", alpha=[0.5])
prostate_glm.train(x=["AGE", "RACE", "PSA", "VOL", "GLEASON"], y="CAPSULE", training_frame=train)

# Show the model
prostate_glm.show()
```

`@solution`

```{python}
prostate = h2o.load_dataset("prostate")
prostate.describe()
# Randomly split the dataset into ~70/30, training/test sets
train, test = prostate.split_frame(ratios=[0.70])

# Convert the response columns to factors (for binary classification problems)
train["CAPSULE"] = train["CAPSULE"].asfactor()
test["CAPSULE"] = test["CAPSULE"].asfactor()

# Build a (classification) GLM
from h2o.estimators import H2OGeneralizedLinearEstimator
prostate_glm = H2OGeneralizedLinearEstimator(family="binomial", alpha=[0.5])
prostate_glm.train(x=["AGE", "RACE", "PSA", "VOL", "GLEASON"], y="CAPSULE", training_frame=train)

# Show the model
prostate_glm.show()
```

`@sct`

```{python}

```


