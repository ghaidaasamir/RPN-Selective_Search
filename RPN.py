import torch
import torchvision.models as models
from torch import nn
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

backbone = nn.Sequential(*list(models.vgg16(pretrained=True).features)[:30])
backbone.out_channels = 512  
anchor_gen = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
rpn_head = RPNHead(in_channels=backbone.out_channels, num_anchors=anchor_gen.num_anchors_per_location()[0])

rpn = RegionProposalNetwork(
    anchor_generator=anchor_gen,
    head=rpn_head,
    fg_iou_thresh=0.8,
    bg_iou_thresh=0.2,
    batch_size_per_image=256,
    positive_fraction=0.5,
    pre_nms_top_n={'training': 1000, 'testing': 500},
    post_nms_top_n={'training': 1000, 'testing': 500},
    nms_thresh=0.6
)

rpn.eval()

# Create a dummy image tensor
# dummy_img = torch.rand(1, 3, 800, 800)  
# Define the image path
# image_path = '/home/ghaidaa/cv/2_2_YOLO/images/food.jpg'
image_path = "YOLO/images/person.jpg"
dummy_img = load_image(image_path)
feature_map = backbone(dummy_img)
features = {'0': feature_map}  
image_list = ImageList(dummy_img, [(800, 800)])

proposals, _ = rpn(image_list, features) 
# print("Proposals:", proposals)

# dummy_img_np = dummy_img.permute(0, 2, 3, 1).numpy()[0]  
# fig, ax = plt.subplots(1, figsize=(12, 12))
# ax.imshow(dummy_img_np)
# for box in proposals[0]:
#     rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)
# plt.show()

dummy_img_np = dummy_img.permute(0, 2, 3, 1).numpy()[0] 
dummy_img_cv = (dummy_img_np * 255).astype(np.uint8)
dummy_img_cv = cv2.cvtColor(dummy_img_cv, cv2.COLOR_RGB2BGR)

output_folder = "YOLO/RPN_output"
for idx, box in enumerate(proposals[0]):

    x1 = int(box[0].item()) if hasattr(box[0], 'item') else int(box[0])
    y1 = int(box[1].item()) if hasattr(box[1], 'item') else int(box[1])
    x2 = int(box[2].item()) if hasattr(box[2], 'item') else int(box[2])
    y2 = int(box[3].item()) if hasattr(box[3], 'item') else int(box[3])
    
    cropped_region = dummy_img_cv[y1:y2, x1:x2]    
    output_filename = os.path.join(output_folder, f"detection_{idx}.jpg")
    
    cv2.imwrite(output_filename, cropped_region)
    print(f"Saved proposal {idx} to {output_filename}")

    # cv2.rectangle(dummy_img_cv, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

# cv2.imwrite("YOLO/RPN_output/detections.jpg", dummy_img_cv)
