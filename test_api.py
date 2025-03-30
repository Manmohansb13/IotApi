import requests

# Define the API URL (update with your actual deployment URL when hosted)
API_URL = "http://127.0.0.1:8000/predict"  # Localhost for testing

# Example input data
test_data = {
    "soil_moisture": 25.0,
    "N": 50.0,
    "P": 30.0,
    "K": 20.0,
    "soil_pH": 6.5,
    "land_size": 2.0,
    "last_crop": "Onion",
    "crop": "Sesamum"
}

# Make the POST request
response = requests.post(API_URL, json=test_data)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
