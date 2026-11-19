// ------ 1. Find the elements on the page ------//
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const dropEmpty = document.getElementById("drop-empty");
const dropSelected = document.getElementById("drop-selected");
const preview = document.getElementById("preview");
const fileName = document.getElementById("file-name");
const errorMessage = document.getElementById("error-message");
const predictBtn = document.getElementById("predict-btn");
const uploadForm = document.getElementById("upload-form");

// ------ 2. Rules for a valid file ------//
const ALLOWED_TYPES = ["image/jpeg", "image/png"];
const MAX_SIZE = 5 * 1024 * 1024; // 5MB in bytes
const BUTTON_TEXT = "Check this leaf";

// ------ 3. Switch the drop zone between "empty" and "photo selected" ------//
function showSelected(isSelected) {
    dropEmpty.hidden = isSelected;
    dropSelected.hidden = !isSelected;
    predictBtn.disabled = !isSelected;
}

// ------ 4. Check a file and update the page ------//
function handleFile(file) {
    errorMessage.textContent = "";
    showSelected(false);

    if (!file) {
        return;
    }

    if (!ALLOWED_TYPES.includes(file.type)) {
        errorMessage.textContent = "Please choose a JPG, JPEG or PNG image.";
        fileInput.value = "";
        return;
    }

    if (file.size > MAX_SIZE) {
        errorMessage.textContent = "This image is larger than 5MB. Please choose a smaller one.";
        fileInput.value = "";
        return;
    }

    // The file is valid: show it and enable the button
    preview.src = URL.createObjectURL(file);
    fileName.textContent = file.name;
    showSelected(true);
}

// ------ 5. When a file is chosen by clicking ------//
fileInput.addEventListener("change", function () {
    handleFile(fileInput.files[0]);
});

// ------ 6. Drag and drop ------//
dropZone.addEventListener("dragover", function (event) {
    event.preventDefault(); // allow dropping here
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", function () {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", function (event) {
    event.preventDefault(); // stop the browser opening the image
    dropZone.classList.remove("dragover");

    const file = event.dataTransfer.files[0];

    // Put the dropped file into the real input, so the form sends it
    const transfer = new DataTransfer();
    if (file) {
        transfer.items.add(file);
    }
    fileInput.files = transfer.files;

    handleFile(file);
});

// ------ 7. Show a loading state while the model runs ------//
uploadForm.addEventListener("submit", function () {
    predictBtn.disabled = true;
    predictBtn.textContent = "Analysing…";
});

// If the user comes back with the Back button, reset the page
window.addEventListener("pageshow", function () {
    predictBtn.textContent = BUTTON_TEXT;
    handleFile(fileInput.files[0]);
});