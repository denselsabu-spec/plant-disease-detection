import requests #Library used to send HTTP requests to APIs

#URL of the fastAPI prediction endpoint running in docker
url = "http://localhost:8000/predict"

#path to the image we want to test
image_path = "test_leaf.png"

try:
    #open the image in binary read mode("rb")
    with open(image_path,"rb") as image:

        #Send a POST request to the API
        #THE image is sent as a file upload with the field name "file"
        response = requests.post(
            url,
            files={"file": image}
        )
        #Print the HTTP status code returned by the API
        #200 = Succcess
        #404 = Not found
        #500 = Server Error 
        print(f"Status Code: {response.status_code}")

        #check if the request was successful
        if response.status_code == 200:
            print("Response received from model: ")

            #Convert JSON response into a python dictionary and print it
            print(response.json())
        else:
            print("API returned an error:")

            #Print the error message sent by the API
            print(response.text)

#Handle errors such as:
# - API not running
# - Image file not found
# - Network connection issues
except Exception as e:
    print("Failed: ", e) 