from huggingface_hub import upload_folder
from huggingface_hub import upload_file


upload_file(
    path_or_fileobj="Tenrec/ctr_data_0.1M.csv",
    path_in_repo="ctr_data_0.1M.csv",
    repo_id="axdyer/tenrec-video-ctr-0.1M",
    repo_type="dataset"
)