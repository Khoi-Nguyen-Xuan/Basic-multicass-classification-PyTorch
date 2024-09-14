<h2> Target of the miniproject</h2> 

Design a basic model in PyTorch to classify 1000 data of 4 classes randomly distributed by ```python sklearn.datasets.make_blobs()```. 

Data to classify are visualized below: 

![Unknown-2](https://github.com/user-attachments/assets/74002d6b-1d98-4742-9bcf-3cf1356bd63f)


<h2>Hyperparameters pick</h2>

* Epochs: 1000
* Loss function : CrossEntropyLoss
* Optimizer : Stochastic Gradient Descent (SGD)
* Learning rate : 0.01
* Layers : 4 layers (with 10 hidden units)
* Activation function : 3 ReLU layers, softmax for the last layer

**The model is constructed as below**: 
 ```python
 class BlobModel(nn.Module):
  def __init__(self, input_features, output_features):
     super().__init__()
     self.linear_stack = nn.Sequential(
         nn.Linear(in_features=input_features, out_features = 10), 
         nn.ReLU(),
         nn.Linear(in_features = 10, out_features=10), 
         nn.ReLU(), 
         nn.Linear(in_features = 10, out_features= 10), 
         nn.ReLU(), 
         nn.Linear(in_features = 10, out_features= output_features), 
     )

  def forward(self,x:torch.Tensor) -> torch.Tensor: 
    return self.linear_stack(x)

 ```

**Training loop**: 
```python
#Training loop

torch.manual_seed(42)
torch.cuda.manual_seed(42)

#device agnostic
X_train, X_test = X_train.to(device), X_test.to(device)
Y_train, Y_test = Y_train.to(device), Y_test.to(device)

my_model = my_model.to(device)

epochs = 1000

for epoch in range(epochs):
  my_model.train() 

  train_logits = my_model(X_train)
  train_pred = torch.argmax(torch.softmax(train_logits, dim = 1), dim = 1)

  loss = loss_function(train_logits, Y_train)
  train_acc = accuracy_fn(Y_train, train_pred)

  optimizer.zero_grad()

  loss.backward()

  optimizer.step() 

  #Evaluate 
  my_model.eval() 

  with torch.inference_mode(): 
    test_logits = my_model(X_test)
    test_pred = torch.argmax(torch.softmax(test_logits, dim = 1), dim = 1)

    test_loss = loss_function(test_logits, Y_test)
    test_acc = accuracy_fn(Y_test, test_pred)

  #Visualize the result every 100 epochs
  if epoch%100 == 0:
    print(f"Epoch: {epoch} | Training loss: {loss:.4f}, Training accuracy: {train_acc:.2f}% | Test loss: {test_loss: .4f}, Test accuracy : {test_acc:.2f}%")

```

<h2> Result </h2>
After 1000 epochs, the model successfully reaches 99.5% in accuracy! The evaluation can be seen in the visualization below: 
<br></br>

![Unknown](https://github.com/user-attachments/assets/f1dcfce0-eebd-4b69-9e73-a5d1a77a46d8)



 
