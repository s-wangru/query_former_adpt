from huggingface_hub import HfApi
from pathlib import Path
import os


def main():
    result_root = Path(os.getenv("RESULT_ARTIFACT_ROOT"))
    api = HfApi()
    api.upload_folder(
        folder_path=str(result_root),
        repo_id="wanshenl/tpch",
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
