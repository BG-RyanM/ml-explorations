# Notes

## Section Two

### General

Main Topics
* Loss Functions
* Optimizers

Neuron representation:
`output = fn(w1 * ip1 + w2 * ip2 + ... + wn * ipn + bias)`

(Can be represented with vector multiply, or matrix-vector multiply when dealing with the whole layer and not just one neuron in it)

Activation functions:
Sigmoid(), Tanh(), or ReLU()

### The Jupyter Notebook

150 flower instances, 80% for training, 20% for validation

Custom module based on nn.Module

`__init__()` method:
creates layers, activation function

`forward()` method:
passes inputs through all layers, leading to output

One `DataLoader` for training set, one for test set. Notice use of "shuffle".

Install pandas with:
`$ conda install -c anaconda pandas`

#### Loss Function
Compares outputs of model to actual ground truth
Objective is to reduce loss score

PyTorch has a whole supply of loss functions, such as `CrossEntropyLoss`
Example: class 0, apple, 0.02 | class 1, mango, 0.88 | class 2, banana, 0.1 --> class 1 is winner

#### Optimizers
Optimization problem: problem of finding best solution from all feasible ones, e.g. best supermarket register to get in line at.

Calculate gradients of loss WRT to model parameters, then update model weights. That's a single iteration.

Provided optimizers in PT: `SGD`, `Adadelta`, `Adam`, `RMSprop`, etc.
Parameters:
* learning rate: step size in direction of lower loss (too big, it might not converge; too small, it might take too long). Must be deduced through experimentation. Active area of research.

#### Training It

Outer loop: iterates through epochs
Inner loop: iterate through all 120 items in training set

Inner loop steps:
* set up inputs and desired outputs: "items", "classes"
* prepare for training
* clear the gradients with zero_grad
* calculate loss
* do backprop to find gradients
* adjust model params based on gradients

#### Outdated Code

Replace `loss.data[0]` with `loss.item()`

### Convert to GPU

(It's basically just adding a `cuda()` call in certain places.)
```
# first replacement
net = IrisNet(4, 100, 50, 3).cuda()

# inner loop
items = Variable(items.cuda())
classes = Variable(classes.cuda())

# last part
outputs = net(Variable(test_items.cuda()))
loss = criterion(outputs, Variable(test_classes.cuda()))
test_loss.append(loss.item())
_, predicted = torch.max(outputs.data, 1)
total = test_classes.size(0)
correct = (predicted == test_classes.cuda()).sum()
test_accuracy.append((100 * correct / total))
```

Don't expect much GPU improvement for lightweight data set compared to running on CPU. Makes more of a difference with image processing.
