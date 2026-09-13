
import streamlit as st
import numpy as np

from ecg_processor import process_ecg
from model_loader import load_model
from inference import predict


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="ECG AI Research Prototype",
    page_icon="❤️",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("❤️ ECG AI Research Prototype")

st.caption(
    "Experimental ECG image processing and AI research tool"
)

st.warning(
    "RESEARCH PROTOTYPE ONLY — This application is not "
    "clinically validated and must not be used to diagnose "
    "patients or make patient-care decisions."
)


# ============================================================
# UPLOAD
# ============================================================

st.header("1. Upload ECG")

uploaded_file = st.file_uploader(
    "Upload a 12-lead ECG image",
    type=["png", "jpg", "jpeg"],
)


if uploaded_file is None:

    st.info(
        "Upload a de-identified ECG image to begin."
    )

    st.stop()


# ============================================================
# SHOW ORIGINAL
# ============================================================

image_bytes = uploaded_file.getvalue()

st.subheader("Original ECG")

st.image(
    image_bytes,
    caption="Uploaded ECG",
    width="stretch",
)


# ============================================================
# PROCESS
# ============================================================

st.header("2. ECG Processing")

if st.button(
    "🔎 PROCESS ECG",
    type="primary",
    width="stretch",
):

    try:

        with st.spinner(
            "Processing ECG image..."
        ):

            result = process_ecg(
                image_bytes
            )

        st.session_state[
            "ecg_result"
        ] = result

        st.success(
            "ECG processing completed."
        )

    except Exception as error:

        st.error(
            f"ECG processing failed: {error}"
        )


# ============================================================
# RESULTS
# ============================================================

if "ecg_result" in st.session_state:

    result = st.session_state[
        "ecg_result"
    ]

    regions = result[
        "regions"
    ]

    waveforms = result[
        "waveforms"
    ]


    # --------------------------------------------------------
    # CLEANED IMAGE
    # --------------------------------------------------------

    st.subheader(
        "Processing Preview"
    )

    st.image(
        result["cleaned"],
        caption="Experimental grid-suppressed image",
        channels="BGR",
        width="stretch",
    )


    # --------------------------------------------------------
    # REGIONS
    # --------------------------------------------------------

    st.subheader(
        "Detected ECG Leads"
    )

    st.write(
        f"Detected regions: **{len(regions)}**"
    )


    columns = st.columns(3)


    for index, region in enumerate(
        regions
    ):

        with columns[index % 3]:

            st.image(
                region["image"],
                caption=region["name"],
                channels="BGR",
                width="stretch",
            )


    # --------------------------------------------------------
    # WAVEFORMS
    # --------------------------------------------------------

    st.subheader(
        "Experimental Waveforms"
    )

    usable = 0


    for index, waveform in enumerate(
        waveforms
    ):

        if waveform is None:
            continue

        if len(waveform) < 2:
            continue

        usable += 1

        st.line_chart(
            waveform,
            height=150,
        )

        st.caption(
            regions[index]["name"]
        )


    st.write(
        f"Usable waveform outputs: "
        f"**{usable}/12**"
    )


    # ========================================================
    # AI
    # ========================================================

    st.header("3. AI Analysis")


    if usable != 12:

        st.warning(
            "AI inference requires 12 usable waveform "
            "outputs. Do not run the model yet."
        )

    else:

        if st.button(
            "🧠 RUN ECG AI",
            type="primary",
            width="stretch",
        ):

            try:

                with st.spinner(
                    "Running experimental AI model..."
                ):

                    model = load_model()

                    results = predict(
                        model,
                        waveforms
                    )


                st.success(
                    "Model inference completed."
                )


                st.subheader(
                    "Research Model Output"
                )


                for item in results:

                    probability = (
                        item["probability"]
                        * 100
                    )

                    st.write(
                        f"**{item['finding']}** — "
                        f"{probability:.1f}%"
                    )

                    st.progress(
                        min(
                            int(probability),
                            100
                        )
                    )


                st.warning(
                    "These scores are experimental model outputs, "
                    "not medical diagnoses. The current model has "
                    "not been clinically validated."
                )


            except Exception as error:

                st.error(
                    f"AI inference failed: {error}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ECG AI Research Prototype — not for clinical use."
)

