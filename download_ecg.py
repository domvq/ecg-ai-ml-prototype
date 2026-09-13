import wfdb

print("Downloading ECG record...")

record = wfdb.rdrecord(
    "100",
    pn_dir="mitdb"
)

print("ECG downloaded!")
print("Sampling rate:", record.fs)
print("Number of samples:", record.sig_len)
print("Number of leads:", record.n_sig)

print("Signal shape:", record.p_signal.shape)