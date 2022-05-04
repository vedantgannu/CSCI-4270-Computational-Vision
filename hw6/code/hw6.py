import sys
#If using Google Colab:
#sys.path.append("/content/drive/MyDrive/Colab Notebooks/CompVis/hw6")
import hw6_datasets_2022 as datasets
from hw6_model_2022 import RCNN
import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2

def area(rect):
    h = rect[3] - rect[1]
    w = rect[2] - rect[0]
    return h * w

#Checking to see if the coordinate is in the bounding box
def coordinate_in_box(rect, coordinates):
    return (rect[0] <= coordinates[0] and rect[2] >= coordinates[0]) and (rect[1] <= coordinates[1] and rect[3] >= coordinates[1])


def iou(rect1, rect2):
    """
    Input: two rectangles
    Output: IOU value, which should be 0 if the rectangles do not overlap.
    """
    #Intersection area / union area
    rect1_area, rect2_area = area(rect1), area(rect2)
    top_x, top_y = max(rect1[0], rect2[0]), max(rect1[1], rect2[1])
    bottom_x, bottom_y = min(rect1[2], rect2[2]), min(rect1[3], rect2[3])
    
    #If the top left and bottom right intersection coordinates are not in the rectangles, there is no intersection
    #Therefore IOU is 0
    if not (coordinate_in_box(rect1, (top_x, top_y)) and coordinate_in_box(rect1, (bottom_x, bottom_y)) \
        and coordinate_in_box(rect2, (top_x, top_y)) and coordinate_in_box(rect2, (bottom_x, bottom_y))):
            return 0
            
    intersection_area = np.abs((bottom_x - top_x) * (bottom_y - top_y))
    union_area = np.abs(rect1_area + rect2_area - intersection_area)
    return intersection_area / union_area


def regression_loss_function(pred_labels, pred_bboxes, gt_boxes, gt_labels):
    #Use the ground truth labels to determine which one of the C bounding boxes will contribute to the loss
    detections_mask = (torch.argmax(pred_labels, dim=1) == gt_labels) & (gt_labels != 0)
    correct_detection_labels = gt_labels[detections_mask]
    #Only compare the bounding boxes for predicted classes that match the ground truth for actual detections
    pred_bboxes_rows = pred_bboxes[detections_mask,:]#4C predicted detection bounding boxes
    class_indices = torch.transpose(torch.cat(((gt_labels[detections_mask] * 4 - 4).view(1,-1), \
                                             (gt_labels[detections_mask] * 4 - 3).view(1,-1), \
                                             (gt_labels[detections_mask] * 4 - 2).view(1,-1), \
                                             (gt_labels[detections_mask] * 4 - 1).view(1,-1)), dim=0), 0,1)
    detection_boxes = pred_bboxes_rows.gather(1, class_indices)
    #Output the average loss for the batch
    criterion = nn.MSELoss()
    loss = criterion(detection_boxes, gt_boxes[detections_mask,:]) if detection_boxes.size()[0] > 0 else torch.Tensor([0]).to("cuda")
    return loss, detection_boxes, gt_boxes[detections_mask,:]

def rescale_bounding_boxes(cand_bboxes_list, detection_bboxes_list):
    cand_bboxes_tensor, detection_bboxes_tensor = torch.as_tensor(cand_bboxes_list), torch.as_tensor(detection_bboxes_list)
    x_scales, y_scales = 224 / (cand_bboxes_tensor[:,[2]] - cand_bboxes_tensor[:,[0]] + 0.1) , 224 / (cand_bboxes_tensor[:,[3]] - cand_bboxes_tensor[:,[1]] + 0.1)
    detection_bboxes_tensor[:,[0, 2]] = (detection_bboxes_tensor[:,[0, 2]] * 224 / x_scales) + cand_bboxes_tensor[:,[0, 0]]
    detection_bboxes_tensor[:,[1, 3]] = (detection_bboxes_tensor[:,[1, 3]] * 224 / y_scales) + cand_bboxes_tensor[:,[1, 1]]
    return detection_bboxes_tensor


def predictions_to_detections(cand_bboxes, predictions_classes, predictions_bboxes, iou_threshold=0.5):
    """
    Output: List of region predictions that are considered to be
    detection results. These are ordered by activation with all class
    0 predictions eliminated, and the non-maximum suppression
    applied.
    """
    #Get all of the non 0 class predictions
    predicted_classes = torch.argmax(predictions_classes, dim=1)
    nonzero_detections_mask = (predicted_classes > 0)
    detected_bboxes_all = predictions_bboxes[nonzero_detections_mask,:]
    detected_cand_bboxes = cand_bboxes[nonzero_detections_mask,:]
    
    detected_classes = predicted_classes[nonzero_detections_mask]
    class_indices = torch.transpose(torch.cat(((detected_classes * 4 - 4).view(1,-1), \
                                             (detected_classes * 4 - 3).view(1,-1), \
                                             (detected_classes * 4 - 2).view(1,-1), \
                                             (detected_classes * 4 - 1).view(1,-1)), dim=0), 0,1)
    detected_bboxes = detected_bboxes_all.gather(1, class_indices)
    activations = predictions_classes[nonzero_detections_mask,:].gather(1, detected_classes.view(-1,1))
    #Highest activation at the front
    sorted_detected_classes_list = detected_classes[torch.flatten(torch.argsort(activations, 0, descending=True)).tolist()].tolist()
    sorted_detected_bboxes_list = detected_bboxes[torch.flatten(torch.argsort(activations, 0, descending=True)).tolist(),:].tolist()
    sorted_cand_bboxes_list = detected_cand_bboxes[torch.flatten(torch.argsort(activations, 0, descending=True)).tolist(),:].tolist()
    
    #Non-maximum suppression
    surpressed = []
    remaining = sorted_detected_bboxes_list.copy()
    keep_bboxes = []
    keep_classes = []
    keep_cand_bboxes = []
    for i in range(len(remaining)):
        rest_predictions = [remaining[j] for j in range(len(remaining)) if remaining[j] != remaining[i] and sorted_detected_classes_list[j] == sorted_detected_classes_list[i]]
        for j in range(len(rest_predictions)):
        #If we can surpress the next prediction:
            if iou(remaining[i], rest_predictions[j]) >= iou_threshold:
                if rest_predictions[j] not in surpressed:
                    surpressed.append(rest_predictions[j])
        if remaining[i] not in surpressed:
            keep_bboxes.append(remaining[i])
            keep_classes.append(sorted_detected_classes_list[i])
            keep_cand_bboxes.append(sorted_cand_bboxes_list[i])
    return keep_bboxes, keep_classes, keep_cand_bboxes



def evaluate(detections_classes, detections_bboxes, gt_detections_classes, gt_detections_bboxes, n=10):
    """
    Returns:
    list of correct detections (bounding boxes and class labels),
    list of incorrect detections (bounding boxes and class labels),
    list of ground truth regions that are missed (bounding boxes and class labels),
    AP@n value.
    """
    
    gt_detections_classes_copy, gt_detections_bboxes_copy = gt_detections_classes.copy(), gt_detections_bboxes.copy()
    regions_size = min(len(gt_detections_bboxes_copy), n)
    B_vector = []
    correct_classes, correct_bboxes = [], []
    incorrect_classes, incorrect_bboxes = [], []
    ground_truthes_classes, ground_truthes_bboxes = [], []
    

    for i in range(len(detections_bboxes[:regions_size])):#Consider at most 'n' detections
        max_IOU = 0
        g_index = None
        for j in range(len(gt_detections_bboxes_copy)):
            if detections_classes[i] == gt_detections_classes_copy[j] and iou(detections_bboxes[i], gt_detections_bboxes_copy[j]) > max_IOU:
                max_IOU = iou(detections_bboxes[i], gt_detections_bboxes_copy[j])
                g_index = j
        if g_index == None or max_IOU < 0.5:
            #If no ground truth value exits for this detection, or if the best ground truth IOU is below 0.5 threshold
            B_vector.append(0)
        else:
            B_vector.append(1)
            correct_classes.append(detections_classes[i])
            correct_bboxes.append(detections_bboxes[i])
            ground_truthes_classes.append(gt_detections_classes[g_index])
            ground_truthes_bboxes.append(gt_detections_bboxes_copy[g_index])
            del gt_detections_classes_copy[g_index]
            del gt_detections_bboxes_copy[g_index]
    B_vector += [0]*(n - regions_size)#Fill B_vector with 0's if number of predicted detections was below 10
    precisions = []
    for i in range(len(B_vector)):
        #Gathering the Precisions at i (the fraction of first i items retrieved that are correct)
        precisions.append(sum(B_vector[0:i+1]) / (i+1))
    average_precision = np.sum(np.array(B_vector)*np.array(precisions)) / min(len(gt_detections_bboxes), n)
    return list(zip(correct_classes, correct_bboxes)),\
        [(detections_classes[i], detections_bboxes[i]) for i in range(len(detections_bboxes)) if detections_bboxes[i] not in correct_bboxes],\
        [(gt_detections_classes[i], gt_detections_bboxes[i]) for i in range(len(gt_detections_bboxes)) if gt_detections_bboxes[i] not in ground_truthes_bboxes],\
        average_precision

def train(training_dataloader, model, class_loss_fn, bounding_box_loss_fn, optimizer, device, hyper_lambda=1, confusion_matrix_bool=False):
    size = len(training_dataloader.dataset)
    batches = len(training_dataloader)
    model.train()
    train_loss_combined = 0
    train_loss1 = 0
    train_loss2 = 0
    class_predictions = []
    gt_classes = []
    for batch, (candidate_region, gt_bbox, gt_label) in enumerate(training_dataloader):
        candidate_region, gt_bbox, gt_label = candidate_region.to(device).float(), gt_bbox.to(device).float()\
                                              ,gt_label.to(device).to(torch.int64)
        # Compute total prediction error (class + bounding box)
        class_pred, bb_box_pred = model(candidate_region)
        loss1 = class_loss_fn(class_pred, gt_label)
        loss2, _, _ = bounding_box_loss_fn(class_pred, bb_box_pred, gt_bbox, gt_label)
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        final_loss = loss1 + (hyper_lambda*loss2)
        train_loss_combined += final_loss.item()
      
        # Backpropagation
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        if batch % 50 == 0:
            final_loss, loss1, loss2, current = final_loss.item(), loss1.item(), loss2.item(), (batch + 1) * len(candidate_region)
            print(f"Batch {batch:>5d}, classification loss: {loss1:>7f}, bbox loss: {loss2:>7f}, combined loss: {final_loss:>7f}  [{current:>5d}/{size:>5d}]")
        if confusion_matrix_bool:
            class_predictions += torch.argmax(class_pred, dim=1).tolist()
            gt_classes += gt_label.tolist()
    train_loss1 /= batches
    train_loss2 /= batches
    train_loss_combined /= batches
    return (train_loss1, train_loss2, train_loss_combined), confusion_matrix(gt_classes, class_predictions) if confusion_matrix_bool else None


def validation(validation_dataloader, model, class_loss_fn, bounding_box_loss_fn, device, hyper_lambda=1, confusion_matrix_bool=False):
    num_batches = len(validation_dataloader)
    model.eval()
    validation_loss_combined = 0
    validation_loss1 = 0
    validation_loss2 = 0
    correct = 0
    gt_detections_count = 0
    class_predictions = []
    gt_classes = []
    total_iou = 0
    with torch.no_grad():
        for (candidate_region, gt_bbox, gt_label) in validation_dataloader:
            candidate_region, gt_bbox, gt_label = candidate_region.to(device).float(), gt_bbox.to(device).float()\
                                              ,gt_label.to(device).to(torch.int64)
            #Compute total prediction error (class + bounding box)
            class_pred, bb_box_pred = model(candidate_region)
            loss1 = class_loss_fn(class_pred, gt_label)
            loss2, pred_detections, gt_detections = bounding_box_loss_fn(class_pred, bb_box_pred, gt_bbox, gt_label)
            validation_loss1 += loss1.item()
            validation_loss2 += loss2.item()
            final_loss = loss1 + (hyper_lambda*loss2)
            validation_loss_combined += final_loss.item()
            
            #Compute the number of correct decisions using classification and bounding box predictions of positive detections
            if pred_detections.size()[0] != 0:#If batch returned any detections
                gt_detections_count += len(gt_detections)
                for pred_box, gt_box in zip(pred_detections, gt_detections):
                    IOU = iou(pred_box.tolist(), gt_box.tolist())
                    if IOU > 0.5:
                        correct += 1
                    total_iou += IOU
            if confusion_matrix_bool:
                class_predictions += torch.argmax(class_pred, dim=1).tolist()
                gt_classes += gt_label.tolist()    
    validation_loss1 /= num_batches
    validation_loss2 /= num_batches
    validation_loss_combined /= num_batches
    return (validation_loss1, validation_loss2, validation_loss_combined, correct/gt_detections_count, total_iou/gt_detections_count),\
                                confusion_matrix(gt_classes, class_predictions) if confusion_matrix_bool else None

def modelTraining(learning_rate, epochs_, device):
    #Classification Block: 512 -> 128, 128 -> 64, 64 -> 5
    #Bounding Box Regression Block -> 512 -> 64, 64 -> 16
    model = RCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epochs = epochs_
    training_losses = []
    validation_losses = []
    confusion_matrix_training, confusion_matrix_validation = None, None
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        print("Training:")
        if t == epochs-1:
            results, confusion_matrix_training = train(training_loader, model, nn.CrossEntropyLoss(),\
                                                regression_loss_function, optimizer, device, hyper_lambda=1, confusion_matrix_bool=True)
        else:
            results, _ = train(training_loader, model, nn.CrossEntropyLoss(),\
                                                regression_loss_function, optimizer, device, hyper_lambda=1, confusion_matrix_bool=False)
        training_losses.append(results)
        print("Avg Training Error:\n\t Classification: {:}, Bounding Box: {:}, Combined: {:}".format(training_losses[t][0],\
                                                    training_losses[t][1], training_losses[t][2]))
        if t == epochs-1:
            results, confusion_matrix_validation = validation(validation_loader, model, nn.CrossEntropyLoss(),\
                                                        regression_loss_function, device, hyper_lambda=1, confusion_matrix_bool=True)
        else:
            results, _ = validation(validation_loader, model, nn.CrossEntropyLoss(),\
                                    regression_loss_function, device, hyper_lambda=1, confusion_matrix_bool=False)
        validation_losses.append(results)
        print("Avg Validation Error:\n\t Classification: {:}, Bounding Box: {:}, Combined: {:}".format(validation_losses[t][0],\
                                                    validation_losses[t][1], validation_losses[t][2]))
        print("Avg Positive Detection Accuracy: {:}\n".format(validation_losses[t][3]))

    print("Training Confusion Matrix")
    training_losses = np.array(training_losses)
    print("Validation Confusion Matrix")
    validation_losses = np.array(validation_losses)
    print(confusion_matrix_training)
    print(confusion_matrix_validation)

    plt.plot(np.linspace(1, epochs, epochs), training_losses[:,[0]], marker='o', color='b', label="Training")
    plt.plot(np.linspace(1, epochs, epochs), validation_losses[:,[0]], marker='o', color='r', label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Classification")
    plt.show()

    plt.plot(np.linspace(1, epochs, epochs), training_losses[:,[1]], marker='o', color='b', label="Training")
    plt.plot(np.linspace(1, epochs, epochs), validation_losses[:,[1]], marker='o', color='r', label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Bounding Box")
    plt.show()

    plt.plot(np.linspace(1, epochs, epochs), training_losses[:,[2]], marker='o', color='b', label="Training")
    plt.plot(np.linspace(1, epochs, epochs), validation_losses[:,[2]], marker='o', color='r', label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Classification + Bounding Box")
    plt.show()
    return model



def modelTesting(best_model, device):
    best_model.eval()
    mAP = 0
    labels = {0: 'nothing', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'person'}
    for index, (image, cand_regions, cand_bboxes, gt_bboxes, gt_classes) in enumerate(testing_loader):
        cand_regions, cand_bboxes, gt_bboxes, gt_classes = cand_regions.to(device).float().squeeze(0), \
                                                            cand_bboxes.to(device).to(torch.int64).squeeze(0), \
                                                            gt_bboxes.to(device).float().squeeze(0), \
                                                            gt_classes.to(device).to(torch.int64).squeeze(0)
        image = image.squeeze(0).numpy()
        pred_classifications, pred_bboxes = best_model(cand_regions)

        #Post processing: Form the actual detections by filtering out 0 class predictions, and to NMS
        filtered_detections_bboxes, filtered_detections_classes, filtered_cand_bboxes =\
                        predictions_to_detections(cand_bboxes, pred_classifications, pred_bboxes, iou_threshold=0.5)
        print(filtered_cand_bboxes)
        if (len(filtered_detections_bboxes) == 0):
            print("No detections found")
            continue
        #Rescale the detection bounding boxes to be in the ground truth coordinate system (original image coordinate system)
        filtered_detections_bboxes = rescale_bounding_boxes(filtered_cand_bboxes, filtered_detections_bboxes).tolist()
        correct_detections, incorrect_detections, missed, AP = evaluate(filtered_detections_classes, filtered_detections_bboxes, gt_classes.tolist(), gt_bboxes.tolist(), n=10)
        print(len(correct_detections))
        print((incorrect_detections))
        print(len(missed))
        print("Average Precision:", AP)
        print()
        mAP += AP
        
        if index in [1, 16, 28, 30, 39, 75, 78, 88, 93, 146]:
            for class_label, box in correct_detections:
                image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
                image = cv2.putText(image, labels[class_label], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 1)

            for class_label, box in incorrect_detections:
                image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 2)
                image = cv2.putText(image, labels[class_label], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,0,255), 1)
            for class_label, box in missed:
                image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,255), 2)
                image = cv2.putText(image, labels[class_label], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,255), 1)
            cv2.imwrite("final_images/{:}.png".format(index), image)
    print("Mean Average Precision:", mAP / len(testing_loader.dataset))





ROOT_google = "/content/drive/MyDrive/Colab Notebooks/CompVis/hw6"
ROOT_aimos = "."
TRAIN_FILES_google, TRAIN_JSON_google = 'small/train', 'small/train.json'
VALID_FILES_google, VALID_JSON_google = 'small/valid', 'small/valid.json'
TRAIN_FILES_aimos, TRAIN_JSON_aimos = 'drive-data/train', 'drive-data/train.json'
VALID_FILES_aimos, VALID_JSON_aimos = 'drive-data/valid', 'drive-data/valid.json'
TEST_FILES_aimos, TEST_JSON_aimos = 'drive-data/test', 'drive-data/test.json'

training_dataset = datasets.HW6Dataset(os.path.join(ROOT_aimos, TRAIN_FILES_aimos), os.path.join(ROOT_aimos, TRAIN_JSON_aimos))
training_loader = DataLoader(training_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
validation_dataset = datasets.HW6Dataset(os.path.join(ROOT_aimos, VALID_FILES_aimos), os.path.join(ROOT_aimos, VALID_JSON_aimos))
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
testing_dataset = datasets.HW6DatasetTest(os.path.join(ROOT_aimos, TEST_FILES_aimos), os.path.join(ROOT_aimos, TEST_JSON_aimos))
testing_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")





if "__main__" == __name__:
    model_15_1e_3 = modelTraining(1e-3, 15, device)
    model_15_1e_5 = modelTraining(1e-5, 15, device)
    model_30_1e_3 = modelTraining(1e-3, 30, device)#Best model
    modelTesting(model_30_1e_3, device)

    
    