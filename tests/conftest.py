import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from term_map.terminology import Terminology  # noqa: E402


def make_term() -> Terminology:
    snomed = pd.DataFrame({
        "code": ["13645005", "282825002", "77547008", "123456", "555555"],
        "description": [
            "COPD - Chronic obstructive pulmonary disease",
            "AF - Paroxysmal atrial fibrillation",
            "Degenerative disc disease",
            "Chronic obstructive pulmonary disease (disorder)",
            "Disc degeneration procedure (procedure)",
        ],
        "description_order": [1, 1, 1, 2, 1],
    })
    icd = pd.DataFrame({
        "code": ["J44.9", "I48.91", "M51.36", "Z87.39"],
        "description": [
            "Chronic obstructive pulmonary disease, unspecified",
            "Unspecified atrial fibrillation",
            "Other intervertebral disc degeneration, lumbar region",
            "Personal history of other musculoskeletal disorders",
        ],
    })
    map_df = pd.DataFrame({
        "snomed_code": ["13645005", "282825002"],
        "icd_code": ["J44.9", "I48.91"],
        "advice": ["ALWAYS J44.9", "ALWAYS I48.91"],
    })
    return Terminology(snomed=snomed, icd=icd, map_df=map_df)


@pytest.fixture()
def term() -> Terminology:
    return make_term()
