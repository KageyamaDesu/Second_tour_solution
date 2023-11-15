Краткое описание всех файлов:
main.py - файл с кодом
image.jpg - анализируемое изображение
start.bat - файл-активатор кода для удобства
runs - в этой папке будут находится изображения с предсказанными рамками и вероятностями (исключительно для визуализации), папка появится сразу после первого предсказания
best.pt - файл с весами
image_predict.txt - итоговый текстовый файл в требуемом формате

Модель обучалась в 10 эпох на 1365 картинках (200 были тестовые). Мы использовали модель YOLO v8. Модель обучалась на процессоре Intel I7-12700 и успела обучиться лишь на 10 эпох. За 10 эпох обучилась до той степени, когда она верно определяет положение циклона, но всё ещё ошибается с его классификацией. В качестве порога определения у нас указано значение 0.5, поэтому на многих изобвжениях, где уверенность модели находится около 0.15, но с верно определённым классом, модель будет выдавать пустой результат. 

Нами было показана, по меньшей мере, принципиальная возможность обучения модели нами.

Возможно, вам будет интересно посмотреть на папку about, в которой находятся файлы, сгенерированные YOLO после обучения модели. Последние 4 изображения иллюстрируют работу модели на тестовой выборке (там, где указаны числа рядом с названием класса находятся результаты предсказания модели). Также там представлены графики, на которых можно заметить, что модель после 10 эпох не нашла локальный минимум и, вероятно, при наличии мощных вычислительных ресурсов мы сможем хорошо решить эту задачу.

Спасибо за внимание! :>

