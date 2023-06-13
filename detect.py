import cv2
import time
from utils import iou
from scipy import spatial
from darkflow.net.build import TFNet
import os
from parse_annotation import parse_annotation



options = {'model': 'cfg/tiny-yolo-voc-3c.cfg',
           'threshold': 0.1,
           }

tfnet = TFNet(options)

pred_bb = []  # predicted bounding box
pred_cls = []  # predicted class
pred_conf = []  # predicted class confidence
annotations = []
RBC = []
WBC = []
Platelets = []

def get_cell_count():
    global annotations
    for i in annotations:
        if 'RBC' in i:
            RBC.append(i['RBC'])
        else:
            RBC.append(0)
        if 'WBC' in i:
            WBC.append(i['WBC'])
        else:
            WBC.append(0)
        if 'Platelets' in i:
            Platelets.append(i['Platelets'])
        else:
            Platelets.append(0)
        


def blood_cell_count(file_name):
    rbc = 0
    wbc = 0
    platelets = 0

    cell = []
    cls = []
    conf = []

    record = []
    tl_ = []
    br_ = []
    iou_ = []
    iou_value = 0

    tic = time.time()
    image = cv2.imread('C:\\Users\\utsav\\Desktop\\Cell Count\\Utsav\\Complete-Blood-Cell-Count-Dataset-master\\Training\\Images\\' + file_name)

    output = tfnet.return_predict(image)

    for prediction in output:
        label = prediction['label']
        confidence = prediction['confidence']
        tl = (prediction['topleft']['x'], prediction['topleft']['y'])
        br = (prediction['bottomright']['x'], prediction['bottomright']['y'])

        if label == 'RBC' and confidence < .5:
            continue
        if label == 'WBC' and confidence < .25:
            continue
        if label == 'Platelets' and confidence < .25:
            continue

        # clearing up overlapped same platelets
        if label == 'Platelets':
            if record:
                tree = spatial.cKDTree(record)
                index = tree.query(tl)[1]
                iou_value = iou(tl + br, tl_[index] + br_[index])
                iou_.append(iou_value)

            if iou_value > 0.1:
                continue

            record.append(tl)
            tl_.append(tl)
            br_.append(br)

        center_x = int((tl[0] + br[0]) / 2)
        center_y = int((tl[1] + br[1]) / 2)
        center = (center_x, center_y)

        if label == 'RBC':
            color = (255, 0, 0)
            rbc = rbc + 1
        if label == 'WBC':
            color = (0, 255, 0)
            wbc = wbc + 1
        if label == 'Platelets':
            color = (0, 0, 255)
            platelets = platelets + 1

        radius = int((br[0] - tl[0]) / 2)
        image = cv2.circle(image, center, radius, color, 2)
        font = cv2.FONT_HERSHEY_COMPLEX
        image = cv2.putText(image, label, (center_x - 15, center_y + 5), font, .5, color, 1)
        cell.append([tl[0], tl[1], br[0], br[1]])

        if label == 'RBC':
            cls.append(0)
        if label == 'WBC':
            cls.append(1)
        if label == 'Platelets':
            cls.append(2)

        conf.append(confidence)

    toc = time.time()
    pred_bb.append(cell)
    pred_cls.append(cls)
    pred_conf.append(conf)
    avg_time = (toc - tic) * 1000
    print('{0:.5}'.format(avg_time), 'ms')

    cv2.imwrite('output/' + file_name, image)
    #cv2.imshow('Total RBC: ' + str(rbc) + ', WBC: ' + str(wbc) + ', Platelets: ' + str(platelets), image)
    count = {"RBC":rbc, "WBC":wbc, "Platelets": platelets}
    return count
    print('Press "ESC" to close . . .')
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def make_data_frame():
    df = pd.DataFrame(columns=['File Number','RBC','WBC','Platelets'])
    df['File Number'] = annotations['File Number']
    df['RBC'] = RBC
    df['WBC'] = WBC
    df['Platelets'] = Platelets

    final_df = df.sort_values(by=['File Number']).reset_index(drop=True)
    Y = final_df[['RBC','WBC','Platelets']]
    Y = Y[['RBC', 'WBC', 'Platelets']].values

    return Y


def load_images_from_folder(folder):
    images_name = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images_name.append(filename)
    return images_name


directory_images = 'C:\\Users\\utsav\\Desktop\\Cell Count\\Utsav\\Complete-Blood-Cell-Count-Dataset-master\\Training\\Images\\'
directory_annotations = 'C:/Users/utsav/Desktop/Cell Count/Utsav/Complete-Blood-Cell-Count-Dataset-master/Training/Annotations/'
#LOAD ALL THE IMAGES FROM THE DIRECTORY
images_name = load_images_from_folder(directory_images)

labels = ['RBC', 'WBC', 'Platelet']

correct_rbc = 0
correct_wbc = 0
correct_plt = 0

total = 0
for i in images_name:
    image_annotation = directory_annotations + i[0:17]+'xml'
    ground_truths, labels = parse_annotation(image_annotation, labels)
    #annotations.append(labels)
    predicted = blood_cell_count(i)
    

    if 'RBC' in labels:
        if predicted['RBC'] == labels['RBC']:
            correct_rbc +=1
    if 'WBC' in labels:
        if predicted['WBC'] == labels['WBC']:
            correct_wbc +=1
    if 'Platelets' in labels:
        if predicted['Platelets'] == labels['Platelets']:
            correct_plt +=1
    total += 1

print(f"Accuracy of RBC: {correct_rbc/total*100:.2f}% WBC: {correct_wbc/total*100:.2f}% PLT: {correct_plt/total*100:.2f}%")


print('All Done!')
