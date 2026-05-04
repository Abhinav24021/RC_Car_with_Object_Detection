import os
import glob

label_dir = r"test_1\labels"
txt_files = glob.glob(os.path.join(label_dir, "*.txt"))

count = 0
for txt in txt_files:
    with open(txt, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            if parts[0] == "0":  # If the label says '0' (Roboflow default for single class)
                parts[0] = "1"   # Change it to '1' (Our model's 'green' class)
            new_lines.append(" ".join(parts) + "\n")
            
    with open(txt, "w") as f:
        f.writelines(new_lines)
    count += 1

print(f"Successfully fixed {count} label files!")
