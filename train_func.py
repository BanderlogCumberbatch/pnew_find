def train_function(model, train_loader, valid_loader, criterion, optimizer, scheduler=None,
                   train_on_gpu=False, n_epochs=n_epochs, save_file='C:/Users/PCunit/Desktop/PnewFind/mymodel.pth'):

    # Изначально установим минимальный loss как бесконечность
    valid_loss_min = np.Inf

    # Переведем модель на GPU, если это необходимо
    if train_on_gpu:
        model = model.cuda()

    # Обучение в течение указанного количества эпох
    for epoch in range(1, n_epochs + 1):
        # Инициализируем потери обучения и валидации
        train_loss = 0.0
        valid_loss = 0.0

        # Обучение модели
        model.train()  # Переключим модель в режим обучения
        for data, target in train_loader:
            # Переместим данные на GPU, если это необходимо
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Обнулим градиенты оптимизатора
            optimizer.zero_grad()

            # Прямой проход: рассчитаем выходные данные
            output = model(data)

            # Рассчитаем убыток
            loss = criterion(output, target)

            # Обратный проход: рассчитаем градиенты
            loss.backward()

            # Обновим веса
            optimizer.step()

            # Накопим потери
            train_loss += loss.item() * data.size(0)  # Умножаем на количество примеров в батче

        # Валидация модели
        model.eval()  # Переключим модель в режим валидации с отключением Dropout и BatchNorm
        correct = 0  # Количество правильных ответов
        total = 0  # Общее количество примеров в валидационном наборе
        with torch.no_grad():  # Отключим вычисление градиентов
            for data, target in valid_loader:
                # Переместим данные на GPU, если это необходимо
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                # Прямой проход: рассчитаем выходные данные
                output = model(data)

                # Получим предсказанные классы
                _, predicted = torch.max(output, 1)  # Получаем индекс предсказанного класса

                # Накопим правильные ответы
                total += target.size(0)
                correct += (predicted == target).sum().item()

        # Рассчитаем точность
        accuracy = 100 * correct / total

        # Выведем информацию об эпохе и точности
        print(f'Epoch {epoch}/{n_epochs}.. Accuracy: {accuracy:.2f}%')

        # Сохраним модель
        #torch.save(model.state_dict(), save_file)  # Сохраним только веса и архитектуру модели
        torch.save(model, save_file)  # Сохраним всю модель, включая веса, архитектуру и состояние оптимизатора
        print(f"Модель сохранена в: {save_file}")

    # Переведем модель на CPU после завершения обучения
    model.to('cpu')

    # Проверяем, существует ли файл перед загрузкой
    if os.path.exists(save_file):
        return torch.load(save_file)  # Возвращаем модель
    else:
        print("Ошибка: файл модели не найден после обучения.")
        return None