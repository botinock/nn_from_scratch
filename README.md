# nn_from_scratch
My Deep Neural Network framework from scratch on numpy.

Custom modules must be inherited from __Module__ and implement methods __forward__ and __backward__.
### Example
```
from nn.dataset import DataSet
from nn.model import Sequential
from nn.linear import Linear
from nn.activation import Sigmoid
from nn.loss import BCE
from nn.optimizer import SGD
from nn.weights import xavier_normal

train_dataset = DataSet(X_train, y_train, batch_size=32, shuffle=True)
test_dataset = DataSet(X_test, y_test, batch_size=len(X_test), shuffle=True)

model = Sequential(
    layers=[       
        Linear(X_train.shape[1], 1, weights_init=xavier_normal),
        Sigmoid()
    ],
    opt=SGD(lr=params['lr']))
loss_fn = BCE()

for data, target in train_dataset:
    output = model(data).squeeze()
    loss = loss_fn(output, target)
    model.backward(loss_fn.backward())
```
