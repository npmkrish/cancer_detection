import os


def get_detect_folders(base_path):
    # store only folders from runs/detect
    folder_list = []

    if not os.path.exists(base_path):
        return folder_list

    for item in os.listdir(base_path):
        full_path = os.path.join(base_path, item)

        if os.path.isdir(full_path):
            folder_list.append({
                "name": item,
                "path": full_path,
                "created_time": os.path.getctime(full_path)
            })

    return folder_list


def find_latest_folder(folder_data):
    # manually find latest folder using loop
    if len(folder_data) == 0:
        return None

    latest = folder_data[0]

    for folder in folder_data:
        if folder["created_time"] > latest["created_time"]:
            latest = folder

    return latest


def get_files_from_folder(folder_path):
    file_collection = []

    for item in os.listdir(folder_path):
        full_path = os.path.join(folder_path, item)

        if os.path.isfile(full_path):
            file_size = os.path.getsize(full_path)

            file_collection.append({
                "file_name": item,
                "file_path": full_path,
                "file_size": file_size
            })

    return file_collection


def main():
    main_path = r"runs/detect"

    folders = get_detect_folders(main_path)

    if not folders:
        print("No directories found.")
        return

    latest_folder = find_latest_folder(folders)
    latest_folder = find_latest_folder(folders)

    print(f"Latest Folder: {latest_folder['name']}")
    print("-" * 10)     #print a separator line for better readability
    print(f"Latest Folder Path: {latest_folder['path']}")

    files = get_files_from_folder(latest_folder["path"])

    if not files:
        print("No files found inside the latest folder.")
        return

    for file_info in files:
        print(f"File Name : {file_info['file_name']}")
        print(f"File Path : {file_info['file_path']}")
        print(f"File Size : {file_info['file_size']} bytes")
        print("-" * 40)
        print(f"File Name : {file_info['file_name']}")
        print(f"File Path : {file_info['file_path']}")
        print(f"File Size : {file_info['file_size']} bytes")
        print("-" * 40)
        



if __name__ == "__main__":
    main()           #main function to execute the script
    