import torch
import torchvision.models as models
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.image_list import ImageList
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

backbone = resnet_fpn_backbone('resnet50', pretrained=True)

anchor_gen = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),) * 5,    # 5 sets of sizes, one for each feature map
    aspect_ratios=((0.5, 1.0, 2.0),) * 5       # 5 sets of aspect ratios
)

rpn_head = RPNHead(in_channels=backbone.out_channels, num_anchors=anchor_gen.num_anchors_per_location()[0])

rpn = RegionProposalNetwork(
    anchor_generator=anchor_gen,
    head=rpn_head,
    fg_iou_thresh=0.7,
    bg_iou_thresh=0.3,
    batch_size_per_image=256,
    positive_fraction=0.5,
    pre_nms_top_n={'training': 2000, 'testing': 1000},
    post_nms_top_n={'training': 1000, 'testing': 300},
    nms_thresh=0.6
)

rpn.eval()

image_path = "YOLO/images/person.jpg"
dummy_img = load_image(image_path)

features = backbone(dummy_img)

image_list = ImageList(dummy_img, [(800, 800)])
proposals, _ = rpn(image_list, features)

dummy_img_np = dummy_img.permute(0, 2, 3, 1).numpy()[0] 
dummy_img_cv = (dummy_img_np * 255).astype(np.uint8)
dummy_img_cv = cv2.cvtColor(dummy_img_cv, cv2.COLOR_RGB2BGR)

output_folder = "YOLO/RPN_output"
os.makedirs(output_folder, exist_ok=True)

for idx, box in enumerate(proposals[0]):
    x1 = int(box[0].item()) if hasattr(box[0], 'item') else int(box[0])
    y1 = int(box[1].item()) if hasattr(box[1], 'item') else int(box[1])
    x2 = int(box[2].item()) if hasattr(box[2], 'item') else int(box[2])
    y2 = int(box[3].item()) if hasattr(box[3], 'item') else int(box[3])
    
    cropped_region = dummy_img_cv[y1:y2, x1:x2]
    output_filename = os.path.join(output_folder, f"detection_{idx}.jpg")
    cv2.imwrite(output_filename, cropped_region)
    print(f"Saved proposal {idx} to {output_filename}")
