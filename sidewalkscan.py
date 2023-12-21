## Pipeline for scanning panorama using coordinates

import streetview
from detecto import core, visualize

## Function for pulling images of panoramas
def pull_image(lat, lon, criteria="close", pth='images/image.jpg'):
    panos = streetview.search_panoramas(lat, lon)

    ## Return closest panorama by date instead of proximity
    if(criteria=='date'):
        panos = [item for item in panos if item.date is not None]
        panos.sort(key=lambda x: x.date, reverse=True)
        
    image = streetview.get_panorama(pano_id=panos[0].pano_id)
    image.save(pth)



## Function to run detection model on saved image
def detect(img_path, model_path):

    model = core.Model.load(model_path, ["Curb Ramp", 'Blocked Ramp', 'Missing Curb Ramp', "Irregular Ramp"])
    image = core.read_image(img_path)
    labels, boxes, scores = model.predict(image)

    ## Print detected curb cuts
    print(str(len(labels))+" crosswalk curbs found, "+str(labels.count('Irregular Ramp'))+" irregular, "+ 
          str(labels.count('Blocked Ramp'))+" blocked, "+str(labels.count('Missing Curb Ramp'))+" missing.")
    
    ## Image with bounding boxes
    visualize.show_labeled_image(image, boxes)




if __name__ == "__main__":
    pull_image(41.50899084742544, -81.60538680853644, 'date', 'images/image.jpg')
    detect('images/image.jpg', 'model_weights.pth')