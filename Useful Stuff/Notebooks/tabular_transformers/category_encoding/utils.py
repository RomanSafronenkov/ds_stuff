import torch
from torch import nn

def evaluate(model, dataloader, loss_fn, device):
    model.eval()  # Переводим модель в режим оценки
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Отключаем вычисление градиентов
        for x_cat, x_num, y in dataloader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            y_pred = model(x_cat, x_num)  # Получаем logits

            # Вычисляем loss
            batch_loss = loss_fn(y_pred, y.squeeze())
            total_loss += batch_loss.item() * y.size(0)

            # Вычисляем accuracy
            _, predicted = torch.max(y_pred, 1)  # Получаем предсказанные классы
            total_correct += (predicted.squeeze() == y.squeeze()).sum().item()  # Считаем количество правильных предсказаний
            total_samples += y.size(0)  # Общее количество примеров

    # Средний loss и accuracy
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy