// File input element
const imageInput =
    document.getElementById("imageInput");

// Image preview element
const preview =
    document.getElementById("preview");

// Preview selected image
imageInput.addEventListener("change", () => {

    const file = imageInput.files[0];

    if (file) {

        preview.src =
            URL.createObjectURL(file);

        preview.style.display =
            "block";
    }
});


// Called when Predict button is clicked
async function predictDisease() {

    const file =
        imageInput.files[0];

    // Ensure image selected
    if (!file) {

        alert(
            "Please select a leaf image first."
        );

        return;
    }

    const loader =
        document.getElementById("loader");

    const resultCard =
        document.getElementById("resultCard");

    const result =
        document.getElementById("result");

    // Show loading animation
    loader.style.display = "block";

    // Hide previous result
    resultCard.style.display = "none";

    // Create multipart form data
    const formData =
        new FormData();

    formData.append(
        "file",
        file
    );

    try {

        // Send image to FastAPI
        const response =
            await fetch("/predict", {

                method: "POST",

                body: formData
            });

        // Convert JSON response
        const data =
            await response.json();

        // Hide loader
        loader.style.display =
            "none";

        // Show prediction
        result.innerHTML = `
            <strong>Disease:</strong>
            ${data.predicted_class}
            <br><br>

            <strong>Confidence:</strong>
            ${data.confidence}%
        `;

        // Display result card
        resultCard.style.display =
            "block";

    }
    catch (error) {

        loader.style.display =
            "none";

        result.innerText =
            "Prediction failed.";

        resultCard.style.display =
            "block";

        console.error(error);
    }
}