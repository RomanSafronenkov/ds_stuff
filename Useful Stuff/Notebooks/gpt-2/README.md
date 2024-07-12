- [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU?si=roYCocmp1-b2nfsP)

## Content:
- [Play.ipynb](./Play.ipynb) - ноутбук, в котором собрана общая информация и прочие мелкие вещи
- [load_pretrained_gpt2.py](./work_through/load_pretrained_gpt2.py) - скрипт, в котором воспроизведена архитектура GPT-2 и реализована функция для погрузки весов обученной модели
- [train_gpt2_vanilla.py](./work_through/train_gpt2_vanilla.py) - скрипт, в котором описан код по обучению GPT-2, продолжение скрипта load_pretrained_gpt2.py, некоторые улучшения архитектуры
- [speeded_up_train_gpt2.py](./work_through/speeded_up_train_gpt2.py) - скрипт, в котором добавлены трюки для ускорения обучения модели (FL32-TF32-BF16, torch.autocast, torch.compile, Flash-Attention with F.scaled_dot_product_attention, красивые числа - степени двойки и как они влияют на скорость обучения)
- [train_optimization_train_gpt2.py](./work_through/train_optimization_train_gpt2.py) - финальный скрипт обучения со всеми фишками

## gpt2_train
- [train.py](./gpt2_train/train.py) - скрипт c обучением gpt-2
- [model.py](./gpt2_train/model.py) - описание модели
- [data](./gpt2_train/data) - директория со скриптами для загрузки данных