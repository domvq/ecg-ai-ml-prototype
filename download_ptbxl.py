
import os
import requests
import pandas as pd


BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"

DATA_DIR = "ptb-xl"

SAMPLE_SIZE = 20


os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# DOWNLOAD METADATA
# ============================================================

print("Downloading PTB-XL metadata...")


metadata_url = (
    f"{BASE_URL}/ptbxl_database.csv"
)

scp_url = (
    f"{BASE_URL}/scp_statements.csv"
)


metadata = pd.read_csv(
    metadata_url
)

scp = pd.read_csv(
    scp_url
)


metadata.to_csv(
    os.path.join(
        DATA_DIR,
        "ptbxl_database.csv"
    ),
    index=False
)


scp.to_csv(
    os.path.join(
        DATA_DIR,
        "scp_statements.csv"
    ),
    index=False
)


print(
    f"Downloaded metadata for "
    f"{len(metadata)} ECG records."
)


# ============================================================
# DOWNLOAD SAMPLE ECG FILES
# ============================================================

print()
print(
    f"Downloading {SAMPLE_SIZE} ECG records..."
)


sample = metadata.head(
    SAMPLE_SIZE
)


def download_file(url, destination):

    response = requests.get(
        url,
        timeout=60
    )

    response.raise_for_status()

    os.makedirs(
        os.path.dirname(destination),
        exist_ok=True
    )

    with open(
        destination,
        "wb"
    ) as file:

        file.write(
            response.content
        )


for number, (_, row) in enumerate(
    sample.iterrows(),
    start=1
):

    filename = row["filename_lr"]

    print(
        f"[{number}/{SAMPLE_SIZE}] "
        f"{filename}"
    )


    # WFDB records consist of two files:
    #
    # .hea = header
    # .dat = ECG signal


    header_url = (
        f"{BASE_URL}/{filename}.hea"
    )

    data_url = (
        f"{BASE_URL}/{filename}.dat"
    )


    header_path = os.path.join(
        DATA_DIR,
        f"{filename}.hea"
    )

    data_path = os.path.join(
        DATA_DIR,
        f"{filename}.dat"
    )


    download_file(
        header_url,
        header_path
    )

    download_file(
        data_url,
        data_path
    )


print()
print("================================")
print("DOWNLOAD COMPLETE")
print("================================")
print()
print(
    f"Downloaded {SAMPLE_SIZE} ECG records."
)
print()
print(
    f"Dataset directory: {DATA_DIR}"
)

