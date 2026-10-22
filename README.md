# MLP_NUMPY

Минимальная реализация MLP на NumPy с автоматическим дифференцированием.

## Что здесь есть

- собственный класс `Tensor_np` с поддержкой backpropagation;
- операции `+`, `*`, `@`, `relu`, `sum` и `bcewithlogits`;
- слой `Linear`;
- слой `BatchNorm1d_np`;
- оптимизатор `SGD`;
- модель `MLP`.

## Цель проекта

Этот репозиторий нужен для обучения и экспериментов с тем, как устроены нейросети и автоматическое дифференцирование "с нуля" на чистом NumPy без использования PyTorch.

## Быстрый пример

```python
import numpy as np
from Tensor_np import Tensor_np
from Linear_np import Linear
from MLP_np import MLP
from SGD_np import SGD

x = Tensor_np(np.random.randn(5, 3).astype(np.float32))
model = MLP(3, 4, 1)
logits = model(x)
print(logits.data.shape)

opt = SGD(model.parameters(), lr=0.01)
```

## Структура

- `Tensor_np.py` — базовый тензор и автодифференциация;
- `Linear_np.py` — линейный слой;
- `BatchNorm1d_np.py` — batch norm;
- `MLP_np.py` — многоуровневая сеть;
- `SGD_np.py` — оптимизатор.

## Автор

Проект создан для изучения работы нейросетей и backpropagation без сторонних библиотек обучения.
