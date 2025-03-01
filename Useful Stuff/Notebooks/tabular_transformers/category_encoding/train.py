import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import time
import logging
from model import CategoryDataset, CategoryEmbeddingModel
from config import config
from tabulate import tabulate  # Для красивого вывода таблицы

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

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

def print_metrics_table(train_metrics, val_metrics, test_metrics):
    """
    Выводит метрики в виде таблицы.
    """
    table = [
        ["Dataset", "Loss", "Accuracy"],
        ["Train", train_metrics[0], train_metrics[1]],
        ["Validation", val_metrics[0], val_metrics[1]],
        ["Test", test_metrics[0], test_metrics[1]]
    ]
    logging.info("\n" + tabulate(table, headers="firstrow", tablefmt="pretty"))

def train_model(train_set, val_set, test_set, cat_cols, num_cols, target, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Вычисляем embedding_dims
    categorical_cardinality = [train_set[col].nunique() + 1 for col in cat_cols]  # +1 для неизвестных категорий
    embedding_dims = [(x, min(50, (x + 1) // 2)) for x in categorical_cardinality]
    logging.info(f"Computed embedding_dims: {embedding_dims}")

    # Инициализация модели, функции потерь и оптимизатора
    model = CategoryEmbeddingModel(embedding_dims=embedding_dims, n_num_cols=len(num_cols)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Инициализация датасетов и даталоадеров
    train_dataset = CategoryDataset(train_set, cat_cols=cat_cols, num_cols=num_cols, target_col=target)
    val_dataset = CategoryDataset(val_set, cat_cols=cat_cols, num_cols=num_cols, target_col=target)
    test_dataset = CategoryDataset(test_set, cat_cols=cat_cols, num_cols=num_cols, target_col=target)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Оценка модели до обучения
    val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn, device)
    logging.info(f'Untrained model scores:\nloss: {val_loss}\naccuracy: {val_accuracy}')

    # Обучение модели
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config["n_epochs"]):
        model.train()
        train_losses = []
        start_time = time.time()

        # Обучение на train_dataloader
        for x_cat, x_num, y in train_dataloader:
            optimizer.zero_grad()

            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)

            y_pred = model(x_cat, x_num)
            batch_loss = loss_fn(y_pred, y.squeeze())
            train_losses.append(batch_loss.item())

            batch_loss.backward()
            optimizer.step()

        # Оценка на val_dataloader
        val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn, device)
        logging.info(f"Epoch: {epoch + 1}. Train loss: {np.mean(train_losses)}. Val loss: {val_loss}. Val accuracy: {val_accuracy}")

        # Ранняя остановка и сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config["best_model_path"])
            logging.info(f"New best model saved with val_loss: {best_val_loss}")
        else:
            epochs_without_improvement += 1
            logging.info(f"No improvement for {epochs_without_improvement} epochs")

        # Прекращаем обучение, если val_loss не улучшается в течение patience эпох
        if epochs_without_improvement >= config["patience"]:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Загрузка лучшей модели
    model.load_state_dict(torch.load(config["best_model_path"]))
    logging.info("Training complete. Best model loaded.")

    # Оценка на тестовых данных
    test_loss, test_accuracy = evaluate(model, test_dataloader, loss_fn, device)
    logging.info(f"Test scores:\nloss: {test_loss}\naccuracy: {test_accuracy}")

    # Итоговые метрики
    train_metrics = evaluate(model, train_dataloader, loss_fn, device)
    val_metrics = evaluate(model, val_dataloader, loss_fn, device)
    test_metrics = (test_loss, test_accuracy)

    # Вывод итоговых метрик в виде таблицы
    print_metrics_table(train_metrics, val_metrics, test_metrics)

if __name__ == "__main__":
    from config import config
    import pandas as pd

    # Загрузка данных
    train_set = pd.read_csv(config["train_data_path"])
    val_set = pd.read_csv(config["val_data_path"])
    test_set = pd.read_csv(config["test_data_path"])

    # Запуск обучения
    train_model(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        cat_cols=config["cat_cols"],
        num_cols=config["num_cols"],
        target=config["target"],
        config=config
    )