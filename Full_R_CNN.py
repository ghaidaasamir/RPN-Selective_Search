import torch
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import nms
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import numpy as np 
import cv2 
import os 

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, 800))
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img

# Initialize the backbone and RPN.
backbone = resnet_fpn_backbone('resnet50', pretrained=True)
anchor_gen = AnchorGenerator(sizes=((32, 64, 128, 256, 512),) * 5, aspect_ratios=((0.5, 1.0, 2.0),) * 5)
rpn_head = RPNHead(in_channels=backbone.out_channels, num_anchors=anchor_gen.num_anchors_per_location()[0])
rpn = RegionProposalNetwork(anchor_generator=anchor_gen, head=rpn_head, fg_iou_thresh=0.7, bg_iou_thresh=0.3, 
                            batch_size_per_image=256, positive_fraction=0.5, pre_nms_top_n={'training': 2000, 'testing': 1000}, 
                            post_nms_top_n={'training': 1000, 'testing': 300}, nms_thresh=0.6)

rpn.eval()

image_path = "YOLO/images/person.jpg"
dummy_img = load_image(image_path)

features = backbone(dummy_img)

image_list = ImageList(dummy_img, [(800, 800)])
proposals, _ = rpn(image_list, features)

def crop_region_from_image(image, box, output_size=(224, 224)):
    _, _, H, W = image.shape
    x1 = max(int(box[0].item()), 0)
    y1 = max(int(box[1].item()), 0)
    x2 = min(int(box[2].item()), W)
    y2 = min(int(box[3].item()), H)
    crop = image[0, :, y1:y2, x1:x2].unsqueeze(0)
    crop_resized = F.interpolate(crop, size=output_size, mode='bilinear', align_corners=False)
    return crop_resized

classifier_model = models.vgg16(pretrained=True)
classifier_model.eval()
fc7_extractor = nn.Sequential(*list(classifier_model.classifier.children())[:5])
fc7_extractor.eval()
svm_classifier = nn.Linear(4096, 2)
svm_classifier.eval()
bbox_regressor = nn.Linear(4096, 4)
bbox_regressor.eval()

output_folder = "YOLO/RPN_full_code"
os.makedirs(output_folder, exist_ok=True)

final_boxes = []
final_scores = []

for idx, proposal in enumerate(proposals[0]):
    # Crop based on the proposal
    region = crop_region_from_image(dummy_img, proposal, output_size=(224, 224))
    
    # Extract features using VGG16's convolutional part, average pool, and fc7 layer
    with torch.no_grad():
        region_conv = classifier_model.features(region)
        region_pooled = classifier_model.avgpool(region_conv)
        region_flat = torch.flatten(region_pooled, 1)
        fc7_feature = fc7_extractor(region_flat)
    
    # Classify using SVM
    with torch.no_grad():
        scores = svm_classifier(fc7_feature)
        probabilities = F.softmax(scores, dim=1)
        object_score = probabilities[0, 1].item()

    if object_score > 0.3:
        with torch.no_grad():
            deltas = bbox_regressor(fc7_feature)
        deltas = deltas[0]
        x1, y1, x2, y2 = proposal
        w = x2 - x1
        h = y2 - y1
        dx, dy, dw, dh = deltas

        refined_x1 = x1 + dx * w
        refined_y1 = y1 + dy * h
        refined_x2 = x2 + dw * w
        refined_y2 = y2 + dh * h

        refined_box = torch.tensor([refined_x1, refined_y1, refined_x2, refined_y2])
        final_boxes.append(refined_box)
        final_scores.append(object_score)

        refined_region = crop_region_from_image(dummy_img, refined_box, output_size=(224, 224))
        refined_region_np = refined_region.squeeze(0).permute(1, 2, 0).numpy()
        refined_region_np = (refined_region_np * 255).astype(np.uint8)  

        # Save the proposal region to the output folder
        output_filename = os.path.join(output_folder, f"detection_{idx}.jpg")
        cv2.imwrite(output_filename, refined_region_np)
        print(f"Saved proposal {idx} to {output_filename}")

nms_threshold = 0.3 
if final_boxes:
    final_boxes = torch.stack(final_boxes)
    final_scores = torch.tensor(final_scores)

    keep_indices = nms(final_boxes, final_scores, nms_threshold)
    final_boxes = final_boxes[keep_indices]
    final_scores = final_scores[keep_indices]

    print(f"Final boxes after NMS: {len(final_boxes)}")
else:
    print("No proposals passed the threshold")
