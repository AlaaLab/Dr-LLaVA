import os
import requests
from huggingface_hub import HfApi, hf_hub_url, HfFolder

# Function to download all files from a model repository
def download_model_files(model_repo_name, local_directory):
    # Create local directory if it does not exist
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    # Instantiate HfApi to interact with the Hugging Face Hub
    api = HfApi()

    # Get the list of all files in the model repository
    model_files = api.list_repo_files(model_repo_name)

    # Loop over each file and download it
    for file in model_files:
        # Get the URL for the specific file
        file_url = hf_hub_url(model_repo_name, filename=file)

        # Set up the headers for authorization if the token exists
        headers = {}
        hf_token = HfFolder.get_token()
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        # Send a GET request to download the file
        response = requests.get(file_url, headers=headers, stream=True)
        response.raise_for_status()

        # Define the local file path
        local_file_path = os.path.join(local_directory, file)

        # Open the file and write to it in chunks
        with open(local_file_path, 'wb') as local_file:
            for chunk in response.iter_content(chunk_size=8192):
                local_file.write(chunk)

        print(f"Downloaded {file} to {local_file_path}")


# Replace 'your_model_name' with the actual model name on Hugging Face
model_name = 'liuhaotian/llava-v1.5-7b'
# Replace 'local_folder_path' with the path to the local directory where you want to download the files
local_folder = 'llava-v1.5-7b'

# Run the function to download the model
download_model_files(model_name, local_folder)
