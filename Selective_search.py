import cv2
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

def selective_search(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ss = selectivesearch.selective_search(img_rgb, scale=500, sigma=0.9, min_size=10)
    # regions = [region['rect'] for region in ss[0] if region['size'] > 2000]
    regions = [region['rect'] for region in ss[1] if region['size'] > 2000]
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img_rgb)

    # cmap = plt.get_cmap('tab20') 
    # print(cmap)
    # colors = [cmap(i) for i in range(len(regions))]

    for rect in regions:
        x, y, w, h = rect
        random_color = (random.random(), random.random(), random.random())
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=random_color, linewidth=1)
        ax.add_patch(rect)

    plt.show()

    return regions

image_path = 'YOLO/images/person.jpg'
proposals = selective_search(image_path)
print("Proposals:", proposals)
