
import cv2
import numpy as np


# Standard 12-lead ECG order for a common
# 3-column × 4-row printed layout.
LEAD_ORDER = [
    "I", "aVR", "V1", "V4",
    "II", "aVL", "V2", "V5",
    "III", "aVF", "V3", "V6",
]


def load_image(image_bytes):
    """
    Convert uploaded image bytes into an OpenCV image.
    """

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Could not read the ECG image."
        )

    return image


def preprocess_image(image):
    """
    Convert ECG image to grayscale and improve
    waveform visibility.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Light denoising.
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Adaptive threshold helps separate the
    # ECG trace from the paper/background.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    return gray, binary


def split_into_leads(image):
    """
    Split a standard 3-column × 4-row ECG
    printout into 12 lead regions.

    This does NOT identify leads automatically.
    It assumes the common printed arrangement.
    """

    height, width = image.shape[:2]

    # Ignore a small outer border.
    margin_x = int(width * 0.02)
    margin_y = int(height * 0.05)

    usable_width = width - (
        2 * margin_x
    )

    usable_height = height - (
        2 * margin_y
    )

    cell_width = usable_width / 3
    cell_height = usable_height / 4

    leads = []

    for row in range(4):

        for column in range(3):

            x1 = int(
                margin_x +
                column * cell_width
            )

            x2 = int(
                margin_x +
                (column + 1) * cell_width
            )

            y1 = int(
                margin_y +
                row * cell_height
            )

            y2 = int(
                margin_y +
                (row + 1) * cell_height
            )

            crop = image[
                y1:y2,
                x1:x2
            ]

            leads.append(
                {
                    "name": LEAD_ORDER[
                        row * 3 + column
                    ],
                    "image": crop,
                }
            )

    return leads


def extract_waveform(
    lead_image
):
    """
    Experimental waveform extraction.

    Finds the darkest trace for each
    horizontal position.

    This is intended for research/testing
    and does not provide calibrated ECG
    measurements.
    """

    gray = cv2.cvtColor(
        lead_image,
        cv2.COLOR_BGR2GRAY
    )

    # Remove a little border.
    h, w = gray.shape

    crop_margin_y = int(
        h * 0.10
    )

    gray = gray[
        crop_margin_y:
        h - crop_margin_y,
        :
    ]

    # Normalize contrast.
    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Dark pixels are potential ECG traces.
    threshold = cv2.threshold(
        gray,
        110,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    waveform = []

    for x in range(
        threshold.shape[1]
    ):

        column = threshold[
            :,
            x
        ]

        pixels = np.where(
            column > 0
        )[0]

        if len(pixels) == 0:

            waveform.append(
                np.nan
            )

        else:

            # Median is more robust than
            # simply taking the first pixel.
            waveform.append(
                float(
                    np.median(pixels)
                )
            )

    waveform = np.asarray(
        waveform,
        dtype=np.float32
    )

    # Interpolate missing values.
    valid = np.isfinite(
        waveform
    )

    if valid.sum() < 2:

        raise ValueError(
            "Could not extract a waveform."
        )

    x = np.arange(
        len(waveform)
    )

    waveform[
        ~valid
    ] = np.interp(
        x[~valid],
        x[valid],
        waveform[valid]
    )

    # Center the signal.
    waveform -= waveform.mean()

    # Normalize amplitude.
    maximum = np.max(
        np.abs(waveform)
    )

    if maximum > 0:

        waveform /= maximum

    return waveform


def preprocess_ecg(image):
    """
    Basic preprocessing for the ECG image.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return gray


def detect_regions(image):
    """
    Divide a standard 3 x 4 ECG printout
    into exactly 12 candidate regions.

    Assumed layout:

        I     aVR   V1
        II    aVL   V2
        III   aVF   V3
        V4    V5    V6
    """

    height, width = image.shape[:2]

    margin_x = int(
        width * 0.02
    )

    margin_y = int(
        height * 0.05
    )

    usable_width = (
        width - 2 * margin_x
    )

    usable_height = (
        height - 2 * margin_y
    )

    cell_width = (
        usable_width / 3
    )

    cell_height = (
        usable_height / 4
    )

    regions = []

    for row in range(4):

        for column in range(3):

            x1 = int(
                margin_x +
                column * cell_width
            )

            x2 = int(
                margin_x +
                (column + 1) *
                cell_width
            )

            y1 = int(
                margin_y +
                row * cell_height
            )

            y2 = int(
                margin_y +
                (row + 1) *
                cell_height
            )

            crop = image[
                y1:y2,
                x1:x2
            ]

            regions.append(
                {
                    "name":
                        LEAD_ORDER[
                            row * 3 + column
                        ],

                    "image":
                        crop,
                }
            )

    return regions



def process_ecg(
    image_bytes
):
    """
    Complete ECG image-processing pipeline.

    Returns exactly the fields expected by app.py.
    """

    # Load original image.
    original = load_image(
        image_bytes
    )

    # Basic preprocessing.
    preprocessed = preprocess_ecg(
        original
    )

    # Suppress likely grid lines.
    cleaned = remove_grid(
        original
    )

    # Create exactly 12 candidate regions.
    regions = detect_regions(
        cleaned
    )

    # Extract one experimental waveform
    # from each region.
    waveforms = []

    for region in regions:

        waveform = extract_waveform(
            region["image"]
        )

        region["waveform"] = waveform

        waveforms.append(
            waveform
        )

    return {
        "image": original,

        "preprocessed":
            preprocessed,

        "cleaned":
            cleaned,

        "regions":
            regions,

        "waveforms":
            waveforms,
    }

# ============================================================
# COMPATIBILITY FUNCTIONS
# ============================================================


def remove_grid(image):
    """
    Suppress likely colored ECG grid lines.
    Experimental image processing only.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower1 = np.array(
        [0, 15, 60],
        dtype=np.uint8
    )

    upper1 = np.array(
        [15, 255, 255],
        dtype=np.uint8
    )

    lower2 = np.array(
        [160, 15, 60],
        dtype=np.uint8
    )

    upper2 = np.array(
        [179, 255, 255],
        dtype=np.uint8
    )

    mask1 = cv2.inRange(
        hsv,
        lower1,
        upper1
    )

    mask2 = cv2.inRange(
        hsv,
        lower2,
        upper2
    )

    mask = cv2.bitwise_or(
        mask1,
        mask2
    )

    cleaned = image.copy()

    cleaned[mask > 0] = (
        255,
        255,
        255
    )

    return cleaned


def process_ecg(image_bytes):
    """Process an uploaded ECG without altering the original image."""

    # Load the uploaded ECG
    original = load_image(image_bytes)

    # Keep the original image unchanged for now.
    cleaned = original.copy()

    # Basic preprocessing
    preprocessed = preprocess_ecg(original)

    # Split the original ECG into 12 candidate regions
    regions = detect_regions(cleaned)

    # Extract experimental waveforms
    waveforms = extract_lead_waveforms(
        cleaned,
        regions
    )

    # Attach each waveform to its region
    for i, waveform in enumerate(waveforms):
        regions[i]["waveform"] = waveform

    return {
        "image": original,
        "preprocessed": preprocessed,
        "cleaned": cleaned,
        "regions": regions,
        "waveforms": waveforms,
    }


def extract_lead_waveforms(image, regions):
    """
    Extract one experimental waveform from each ECG region.
    """

    waveforms = []

    for region in regions:
        waveform = extract_waveform(
            region["image"]
        )

        waveforms.append(
            waveform
        )

    return waveforms




