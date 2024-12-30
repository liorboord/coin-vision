from google_images_search import GoogleImagesSearch
import os
import requests

# Enter your API key and CSE ID here
API_KEY = 'AIzaSyCb0BksEQCObuqwhEn3MlboZXnNbMhzKbY'
CSE_ID = '937f2d90ffb354f52'
DOWNLOAD_FOLDER = "../static/coin_images_raw_google"

# Initialize Google Images Search with API key and CSE ID
gis = GoogleImagesSearch(API_KEY, CSE_ID)

def google_image_search(query, num_results=10, download_folder=DOWNLOAD_FOLDER):
    # Create the download folder if it doesn't exist
    os.makedirs(download_folder, exist_ok=True)

    # Set up search parameters
    search_params = {
        'q': query,
        'num': num_results,
        'fileType': 'jpg',      # Only download .jpg files
        'safe': 'medium',       # Safe search setting
        'imgType': 'photo',     # Only photos, can be adjusted if needed
    }

    gis.search(search_params=search_params)

    # Download each image
    for i, image in enumerate(gis.results()):
        try:
            image_url = image.url
            response = requests.get(image_url, stream=True)
            response.raise_for_status()  # Check if the request was successful

            # Define image filename and save path
            image_filename = os.path.join(download_folder, f"coin_5_{i + 1}.jpg")

            # Write the image data to a file
            with open(image_filename, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

            print(f"Downloaded {image_filename}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download image {i + 1}: {e}")

# Example usage
query = "מטבע 5 שקל"
google_image_search(query, num_results=100)