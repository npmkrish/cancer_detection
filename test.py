from ultralytics import YOLO
model=YOLO('best5.pt')

results=model.predict(source='static/images/b8a2aa251bdf4a0b8a376137b6d7f8d9_image.jpeg', save=True, save_txt=True, conf=0.5, iou=0.5, device='0', show=True)
print(results)  

def plot_results(results):
    for result in results:
        plt.figure(figsize=(10, 6))
        plt.imshow(result.plot())
        plt.title('YOLOv8 Detection Results')
        plt.axis('off')
        plt.show()
def get_detect_folders(main_path):
    folder_list = []

    for item in os.listdir(main_path):
        full_path = os.path.join(main_path, item)

        if os.path.isdir(full_path):
            folder_list.append({
                "name": item,
                "path": full_path,
                "created_time": os.path.getctime(full_path)
            })
def get_detect_folders(base_path):
    full_path = os.path.join(base_path, "detect")
     

    return folder_list            
    

    

#//!!