import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from paddleocr import PaddleOCR
from PIL import Image
from ultralytics import YOLO

# Set page config
st.set_page_config(page_title="Medical Image OCR", page_icon="🏥", layout="wide")

# Initialize lock for thread safety
lock = threading.Lock()


# Define paths for models
@st.cache_resource
def load_models():
    det_model_dir = "models/det"
    rec_model_dir = "models/rec"
    cls_model_dir = "models/cls"

    # Initialize OCR
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        det_model_dir=det_model_dir,
        rec_model_dir=rec_model_dir,
        cls_model_dir=cls_model_dir,
        show_log=False,
    )

    # Initialize YOLO model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("best.pt").to(device)

    return ocr, model


# Gold dataset for text correction
gold_dataset = [
    "RT UT ART",
    "G SAC",
    "RT OV",
    "LT OV",
    "LEFT KIDNEY",
    "RIGHT KIDNEY",
    "LT UT ART",
    "Lt Ut A PS",
    "Umb A PS",
    "Umb A HR",
    "Umb A RI",
    "Umb A PI",
    "Umb A ED",
    "MCA PS",
    "MCA RI",
    "MCA PI",
    "RT KIDNEY",
    "LT KIDNEY",
    "RT OVARY",
    "LT OVARY",
    "GALL BLADDER",
    "Lt Ov Vol",
    "EDD 01/01/2025",
    "RETROCECAL APPENDICITIS",
    "NUCHAL LUCENCY",
    "URINARY BLADDER",
    "CORD INSERTION",
    "PLACENTA FUNDAL GRADE I",
    "BREECH SPINE POSTERIOR",
    "PLACENTA POSTERIOR GRADE II",
    "LA Diam",
    "LEFT OVARY",
    "RIGHT OVARY",
    "LOWER UR",
]

# Define a path to a static sample image
SAMPLE_IMAGE_PATH = "test-images/image7.jpg"


# Helper Classes and Functions
class ImageProcessor(threading.Thread):
    def __init__(self, ocr, image_path):
        super().__init__()
        self.ocr = ocr
        self.image_path = image_path
        self.result = None

    def run(self):
        try:
            with lock:
                result = self.ocr.ocr(self.image_path, cls=True)
                if result and result[0]:
                    self.result = result[0]
                else:
                    self.result = []
        except Exception as e:
            self.result = []
            st.error(f"OCR processing error: {str(e)}")


def parse_measurement(text):
    """Parse medical measurement text by splitting on various medical patterns."""
    patterns = [
        r"(-?\d+\.\d+|-?\d+)(?:\(\d+\))?\s*(mmHg|mm|cm/s|cm3|cm|bpm|ml|kPa)|(\d+w\d+d)|(-?\d+\.\d+|-?\d+)(?:\(\d+\))?\s*(%)|(?!.*%)(-?\d+\.\d+|-?\d+)(?:\(\d+\))?",
        r"(\d+w\d+d)",
        r"(\d{2}/\d{2}/\d{4})",
        r"(-?\d+\.\d+|-?\d+)(?:\(\d+\))?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            measurement = text[match.start() : match.end()]
            before_measurement = text[: match.start()].strip()
            after_measurement = text[match.end() :].strip()

            result = []
            if before_measurement:
                result.append(before_measurement)
            result.append(measurement)
            if after_measurement:
                result.extend(after_measurement.split())

            return _enforce_max_length(result)

    return _enforce_max_length(text.split())


def _enforce_max_length(lst):
    """Ensure list has maximum 5 elements."""
    if len(lst) <= 5:
        return lst
    result = lst[:4]
    result.append(" ".join(lst[4:]))
    return result


def format_data(text: str, table_data: list) -> list:
    """Format text and table data according to specific rules."""
    result = [None] * 6

    if text == "":
        result[0] = None
        for i, value in enumerate(table_data):
            if i < len(result) - 1:
                result[i + 1] = value
    elif not table_data:
        result[0] = text
    else:
        result[0] = text
        if table_data:
            result[1] = f"{table_data[0]}"
            for i, value in enumerate(table_data[1:], start=2):
                if i < len(result):
                    result[i] = value

    return result


def post_process_text(ocr_result):
    """Process OCR result to extract text and bounding box information."""
    processed_text = []
    for line in ocr_result:
        text = line[1][0]
        bbox = line[0]
        processed_text.append((text, bbox))
    return processed_text


def arrange_text(processed_text):
    """Arrange text by vertical position and sort horizontally within each line."""
    sorted_text = sorted(processed_text, key=lambda x: x[1][0][1])

    lines = []
    current_line = []
    prev_y = None
    line_height_threshold = 10

    for text, bbox in sorted_text:
        current_y = bbox[0][1]
        if prev_y is None or abs(current_y - prev_y) <= line_height_threshold:
            current_line.append((text, bbox))
        else:
            lines.append(current_line)
            current_line = [(text, bbox)]
        prev_y = current_y

    if current_line:
        lines.append(current_line)

    arranged_lines = []
    for line in lines:
        sorted_line = sorted(line, key=lambda x: x[1][0][0])
        arranged_lines.append(" ".join(text for text, _ in sorted_line))
    return "\n".join(arranged_lines)


def normalize_text(text):
    """Normalize text for comparison by removing non-alphanumeric characters."""
    return "".join(c.upper() for c in text if c.isalnum())


def find_match_in_gold_dataset(text, gold_dataset):
    """Find matching text in gold dataset based on normalized text."""
    normalized_text = normalize_text(text)
    for gold_item in gold_dataset:
        if normalize_text(gold_item) == normalized_text:
            return gold_item
    return None


def correct_spacing(text, gold_dataset):
    """Correct spacing and common OCR errors in medical text."""
    unit_pattern = r"(\d+(\.\d+)?(\(\d+\))?)(?=\s?(cm|cm/s|bpm|mm|kPa))"
    float_pattern = r"(?<!\S)([A-Za-z]+)\s*?(\d+\.\d+)"

    # Common medical term corrections
    text = re.sub(r"MV\s?E\s?Vel", "MV E VEL", text, flags=re.IGNORECASE)
    text = re.sub(r"MV\s?A\s?Ve", "MV A VEL", text, flags=re.IGNORECASE)
    text = re.sub(r"MV\s?E\s?PG", "MV E PG", text, flags=re.IGNORECASE)
    text = re.sub(r"MV\s?A\s?PG", "MV A PG", text, flags=re.IGNORECASE)
    text = re.sub(r"MVE/A", "MV E/A", text, flags=re.IGNORECASE)
    text = re.sub(r"RT\s?LOWERUR\s?CAL", "RT LOWER UR CAL", text, flags=re.IGNORECASE)
    text = re.sub(r"AVVmax", "AV Vmax", text, flags=re.IGNORECASE)
    text = re.sub(r"AVPGmax", "AV PGmax", text, flags=re.IGNORECASE)
    text = re.sub(r"LOWER UR", "LOWER UR", text, flags=re.IGNORECASE)
    text = re.sub(r"POSTVOID BLADDER", "POST VOID BLADDER", text, flags=re.IGNORECASE)
    text = re.sub(r"LOWERLIMBS", "LOWER LIMBS", text, flags=re.IGNORECASE)
    text = re.sub(r"RTCFV", "RT CFV", text, flags=re.IGNORECASE)
    text = re.sub(r"RTSFVP", "RT SFV P", text, flags=re.IGNORECASE)
    text = re.sub(r"RT SFVM", "RT SFV M", text, flags=re.IGNORECASE)
    text = re.sub(r"RT Pop", "RT Pop V", text, flags=re.IGNORECASE)
    text = re.sub(r"RTAT", "RT ATV", text, flags=re.IGNORECASE)
    text = re.sub(r"RTSFVADD", "RT SFV ADD", text, flags=re.IGNORECASE)
    text = re.sub(r"RTPop", "RT Pop V", text, flags=re.IGNORECASE)
    text = re.sub(r"RTADN", "RT ADN", text, flags=re.IGNORECASE)
    text = re.sub(r"LTADN", "LT ADN", text, flags=re.IGNORECASE)
    text = re.sub(r"(Lt\s+Ov-W)(\d)", r"Lt Ov-W \2", text)
    text = re.sub(r"(Lt\s+Oy-L)", r"Lt Ov-L", text, flags=re.IGNORECASE)
    text = re.sub(r"LIVER FATT", r"LIVER FATTY", text, flags=re.IGNORECASE)

    # Convert 'c' patterns to 'd'
    text = re.sub(r"(\d+w\d+)c", r"\1d", text)
    # Add space after pattern if there isn't one
    text = re.sub(r"(\d+w\d+d)([^\s])", r"\1 \2", text)

    words = text.split()
    corrected_words = []
    i = 0

    while i < len(words):
        words[i] = re.sub(unit_pattern, r"\1 ", words[i])
        words[i] = re.sub(float_pattern, r"\1 \2", words[i])

        for j in range(3, 0, -1):
            if i + j > len(words):
                continue
            combined = "".join(words[i : i + j])
            match = find_match_in_gold_dataset(combined, gold_dataset)
            if match:
                corrected_words.append(match)
                i += j
                break
        else:
            corrected_words.append(words[i])
            i += 1

    return " ".join(corrected_words)


def process_single_image(ocr, image_path):
    """Process a single image with OCR in a separate thread."""
    processor = ImageProcessor(ocr, image_path)
    processor.start()
    processor.join()
    return image_path, processor.result


def predict_and_process_image(model, ocr, image_path):
    """Predict regions in image and process them with OCR."""
    with st.spinner("Processing image with YOLO..."):
        # Create temporary directory for cropped images
        temp_dir = tempfile.mkdtemp()

        # Get image name
        base_name = os.path.basename(image_path).split(".")[0]

        # Run YOLO model prediction
        results = model.predict(image_path, verbose=False)
        image = cv2.imread(image_path)

        # Process and save cropped images
        saved_images = []
        detected_classes = []
        for i, result in enumerate(results):
            for j, box in enumerate(result.boxes.xyxy):
                xmin, ymin, xmax, ymax = map(int, box)
                img = image[ymin:ymax, xmin:xmax]
                name = os.path.join(temp_dir, f"{base_name}_crop{i}_{j}.jpg")
                cv2.imwrite(name, img)
                saved_images.append(name)
                detected_classes.append(model.names[int(result.boxes.cls[j])])

    table_data = []
    text_data = []

    # Process each cropped image with OCR
    with st.spinner("Extracting text from images..."):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_image = {
                executor.submit(process_single_image, ocr, img_path): img_path
                for img_path in saved_images
            }

            for i, future in enumerate(as_completed(future_to_image)):
                img_path = future_to_image[future]
                ocr_result = future.result()[1]
                if ocr_result:
                    processed_text = post_process_text(ocr_result)
                    arranged_text = arrange_text(processed_text)
                    lines = arranged_text.split("\n")

                    for line in lines:
                        corrected_line = correct_spacing(line, gold_dataset)
                        idx = saved_images.index(img_path)
                        if (
                            idx < len(detected_classes)
                            and detected_classes[idx].lower() == "table"
                        ):
                            table_data.append(corrected_line)
                        else:
                            text_data.append(corrected_line)

    return text_data, table_data


# Streamlit App
def main():
    st.title("Medical Image OCR Processing")
    st.write("Upload a medical image to extract text and tabular data")

    # Load models with spinner only, no success message
    with st.spinner("Loading models..."):
        ocr, model = load_models()

    # Image selection method
    image_source = st.radio(
        "Select image source", ["Upload a file", "Use sample image"]
    )

    image_path = None

    if image_source == "Upload a file":
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a medical image file", type=["jpg", "jpeg", "png", "bmp", "dcm"]
        )

        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                if uploaded_file.name.lower().endswith(".dcm"):
                    # Convert DICOM to JPG if needed
                    try:
                        import pydicom

                        dicom_data = pydicom.dcmread(uploaded_file)
                        pixel_array = dicom_data.pixel_array
                        normalized_image = (
                            (pixel_array - np.min(pixel_array))
                            / (np.max(pixel_array) - np.min(pixel_array))
                            * 255
                        )
                        normalized_image = normalized_image.astype(np.uint8)
                        image = Image.fromarray(normalized_image)
                        image.save(tmp_file.name, "JPEG")
                    except ImportError:
                        st.error("pydicom module is required for DICOM files")
                        return
                else:
                    # Save other image formats directly
                    file_bytes = np.asarray(
                        bytearray(uploaded_file.read()), dtype=np.uint8
                    )
                    image = cv2.imdecode(file_bytes, 1)
                    cv2.imwrite(tmp_file.name, image)

                image_path = tmp_file.name
    else:
        # Use sample image
        if os.path.exists(SAMPLE_IMAGE_PATH):
            image_path = SAMPLE_IMAGE_PATH
            # st.info(f"Using sample image from: {SAMPLE_IMAGE_PATH}")
        else:
            st.error(f"Sample image not found at: {SAMPLE_IMAGE_PATH}")
            st.info(
                "Please make sure to place a sample image file named 'sample_medical_image.jpg' in the same directory as this script."
            )
            return

    if image_path:
        # Display the image
        st.subheader("Image")
        st.image(image_path, width=600)

        # Process button
        if st.button("Process Image"):
            # Process the image
            text_data, table_data = predict_and_process_image(model, ocr, image_path)

            # Display combined results
            # st.subheader("Extracted Data")
            st.markdown("### Extracted Data", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Text Data", unsafe_allow_html=True)
                if text_data:
                    for i, text in enumerate(text_data):
                        st.write(f"{i+1}. {text}")
                else:
                    st.info("No text data extracted")

            with col2:
                st.markdown("### Table Data", unsafe_allow_html=True)
                if table_data:
                    for i, data in enumerate(table_data):
                        st.write(f"{i+1}. {data}")
                else:
                    st.info("No table data extracted")

            # Create a combined text output for download
            combined_text = "TEXT DATA:\n\n"
            for text in text_data:
                combined_text += f"{text}\n"

            combined_text += "\n\nTABLE DATA:\n\n"
            for data in table_data:
                combined_text += f"{data}\n"

            # Add download button for the combined text
            st.download_button(
                label="Download Results as Text",
                data=combined_text,
                file_name="medical_image_ocr_results.txt",
                mime="text/plain",
            )

            # Clean up the temporary file if it was created
            if image_source == "Upload a file" and image_path:
                try:
                    os.unlink(image_path)
                except:
                    pass


if __name__ == "__main__":
    main()
