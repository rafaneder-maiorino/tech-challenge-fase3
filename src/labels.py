"""Label vocabulary shared by the training pipeline and the serving API.

Kept dependency-free on purpose: the API container needs these constants but
must not pull in pandas, pyarrow or the Hugging Face client to get them.
"""

from __future__ import annotations

TEXT_COLUMN = "medical_abstract"
LABEL_COLUMN = "condition_label"

CONDITION_NAMES = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}
