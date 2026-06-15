# Getting model architecture

Let this be example of our model:
```python
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.fc(x))

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 8)
        self.blocks = nn.ModuleList([Block(8) for _ in range(2)])
        self.head = nn.Linear(8, 2)
    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)

model = TinyNet()
```


## 1.  `print()`

You can get model by simply using print functionality and you would get something like this:

```
TinyNet(
  (embed): Linear(in_features=4, out_features=8, bias=True)
  (blocks): ModuleList(
    (0-1): 2 x Block(
      (fc): Linear(in_features=8, out_features=8, bias=True)
      (act): ReLU()
    )
  )
  (head): Linear(in_features=8, out_features=2, bias=True)
)
```

## 2. `named_modules()`

```
             TinyNet
embed        Linear
blocks       ModuleList
blocks.0     Block
blocks.0.fc  Linear
blocks.0.act ReLU
blocks.1     Block
blocks.1.fc  Linear
blocks.1.act ReLU
head         Linear
```

## 3. `named_parameters()`

Note ReLU never appears - no params - and bias rows show up separately:

```
embed.weight        (8, 4)   32
embed.bias          (8,)      8
blocks.0.fc.weight  (8, 8)   64
blocks.0.fc.bias    (8,)      8
blocks.1.fc.weight  (8, 8)   64
blocks.1.fc.bias    (8,)      8
head.weight         (2, 8)   16
head.bias           (2,)      2

Total: 202
```
