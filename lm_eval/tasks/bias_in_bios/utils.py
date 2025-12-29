import datasets

PROFESSIONS_CANON = [
    "accountant",
    "architect",
    "attorney",
    "chiropractor",
    "comedian",
    "composer",
    "dentist",
    "dietitian",
    "dj",
    "filmmaker",
    "interior_designer",
    "journalist",
    "model",
    "nurse",
    "painter",
    "paralegal",
    "pastor",
    "personal_trainer",
    "photographer",
    "physician",
    "poet",
    "professor",
    "psychologist",
    "rapper",
    "software_engineer",
    "surgeon",
    "teacher",
    "yoga_teacher",
]

PROFESSIONS_TEXT = [p.replace("_", " ") for p in PROFESSIONS_CANON]


def process_bias_in_bios(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        query = (
            "Read the following biography and identify the person's profession.\n\n"
            f"Biography:\n{doc['hard_text']}\n\n"
            "Profession:"
        )

        raw_profession = doc["profession"]

        if isinstance(raw_profession, int):
            if raw_profession > 27:
                gold_idx = raw_profession - 1
            else:
                gold_idx = raw_profession
        else:
            clean_prof = str(raw_profession).strip().lower().replace(" ", "_")
            gold_idx = PROFESSIONS_CANON.index(clean_prof)

        return {
            "query": query,
            "choices": PROFESSIONS_TEXT,
            "gold": gold_idx,
            "gender": int(doc["gender"])
        }

    return dataset.map(_process_doc)