// ------ 1. Find the elements on the page ------*/
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const preview = document.getElementById("preview");
const fileName = document.getElementById("file-name")
const errorMessage = document.getElementById("error-message");
const predictBtn = document.getElementById("predict-btn");

// ------ 2. Rules for  valid file ------*/
const ALLOWED_TYPES = ["image/jpeg", "image/png"];
const MAX_SIZE = 5 * 1024 * 1024; // 5MB BYTES

// ------ 3. Check a file and update the page ------*/
function handleFile(file) {
    //Reset anything left over from a previous file
    errorMessage.textContent = "";
    preview.hidden = true;
    fileName.textContent = "";
    predictBtn.disabled = true;

    if (!file) {
        return;
    }

    if (!ALLOWED_TYPES.includes(file.type)) {
        errorMessage.textContent = "Please choose a JPG, JPEG, or PNG image.";
        fileInput.value = "";
        return;
    }

    if (file.size > MAX_SIZE) {
        errorMessage.textContent = "This image is larger than 5MB. Please choose a smaller one.";
        fileInput.value = "";
        return;
    }

    //The file is valid: show it and enable the button
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    fileName.textContent = file.name;
    predictBtn.disabled = false;
}

// ------ 4. When a file is chosen by clicking -----//
fileInput.addEventListener("change", function() {
    handleFile(fileInput.files[0]);
});

//------ 5. Drag and Drop ------//
dropZone.addEventListener("dragover", function (event) {
    event.preventDefault(); // allow dropping here
    dropZone.classList.add("dragover"); //green highlight
});

dropZone.addEventListener("dragleave", function () {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", function(event) {
    event.preventDefault();  //stop the browser opening the image
    dropZone.classList.remove("dragover");

const file = event.dataTransfer.files[0];

//Put the dropped file into the real input, so the form send it
const transfer = new DataTransfer();
if (file) {
    transfer.items.add(file);
}
fileInput.files = transfer.files;

handleFile(file);

});