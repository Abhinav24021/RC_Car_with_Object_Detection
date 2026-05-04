import xml.etree.ElementTree as ET
import os
import glob

# Your exact classes from class.txt
CLASSES = ["red", "yellow", "green", "off", "wait_on"]

def convert_to_yolo(size, box):
    """Converts pixel coordinates to YOLO percentage coordinates"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x_center * dw, y_center * dh, w * dw, h * dh)

def process_folder(xml_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    xml_files = glob.glob(os.path.join(xml_folder, '*.xml'))
    
    if not xml_files:
        print(f"⚠️ No XML files found in {xml_folder}!")
        return

    print(f"Processing {len(xml_files)} files in {xml_folder}...")
    
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        # Create the new .txt filename
        filename = os.path.basename(xml_file).replace('.xml', '.txt')
        out_file_path = os.path.join(output_folder, filename)
        
        with open(out_file_path, 'w') as out_file:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                
                # Skip any labels we don't care about
                if cls not in CLASSES:
                    continue
                    
                cls_id = CLASSES.index(cls)
                xmlbox = obj.find('bndbox')
                
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                
                bb = convert_to_yolo((w, h), b)
                # Write: Class_ID X_Center Y_Center Width Height
                out_file.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

print("--- Starting YOLO Conversion ---")
# This converts your training data
process_folder(r'C:\Users\jhasa\OneDrive\Desktop\tf lt\train_annotations', r'C:\Users\jhasa\OneDrive\Desktop\tf lt\train_labels')
# This converts your validation data
process_folder(r'C:\Users\jhasa\OneDrive\Desktop\tf lt\valid_annotations', r'C:\Users\jhasa\OneDrive\Desktop\tf lt\valid_labels')
print("--- Conversion Complete! ---")