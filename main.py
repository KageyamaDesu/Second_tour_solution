#!pip3 install ultralytics
from ultralytics import YOLO
import pandas as pd

model = YOLO('./best.pt')
results = model.predict('./image.jpg', imgsz=640, conf=0.5, verbose=False,
                        save=True)  # save - сохранение результата, verbose - вывод информации об этом(False, чтобы не засорять)
for result in results:  # conf - порог обнаружения, imgsz - размер входного изображения
    boxes = result.boxes.data
    path = result.path
if len(boxes) == 0:
    print('None')
    exit()

# Здесь просходит адаптация данных под требуемый формат
out = pd.DataFrame(boxes).astype("float").drop([4], axis=1)
out = out.rename(columns={0: 'x1', 1: 'y1', 2: 'x2', 3: 'y2', 5: 'class'})

nums = [0, 1, 2, 3, 4, 5, 6]
names = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7']
x1, x2, y1, y2 = out['x1'], out['x2'], out['y1'], out['y2']

out['x1'] = (x1 + x2) / 2
out['y1'] = (y1 + y2) / 2
out['x2'] = (x2 - x1) * 2
out['y2'] = (y2 - y1) * 2

out = out[['class', 'x1', 'y1', 'x2', 'y2']].round(0).astype('int')
txtfile = open("image_predict.txt", "w+")
for i in range(7):
    out = out.replace(nums[i], names[i])
for i in range(len(out)):
    a = out.iloc[i].tolist()
    txtfile.write(", ".join(str(x) for x in a)+"\n")
