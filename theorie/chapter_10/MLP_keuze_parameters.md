# Overzicht MLP's

## Regression MLP's

![Alt text](./images/mlp_overview.png)

Typically MSE as loss function but MAE is preferred if there are many outliers.

Huber Loss is a combination of MSE and MAE, it is quadratic for small erros and linear for large erros.

---

## Classification MLP's

![Alt text](./images/mlp_classification_overview.png)

For output activation: logisitc = sigmoid

### Softmax activation function

- typically used for multiclass classification

- computation:
    1. exp z = e^z for each z => (e^z1, e^z2, ..., e^zk) => makes all values positive
    2. Devide each value by the sum of all values => (e^z1 / sum, e^z2 / sum, ..., e^zk / sum) => all values between 0 and 1 and sum up to 1
    3. Example:

        ![Alt text](./images/softmax_excel_example.png)
