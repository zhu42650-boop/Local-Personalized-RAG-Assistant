import re
from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SECTION_PATTERN = re.compile(
    r"^\s*(abstract|introduction|related work|background|methods?|methodology|"
    r"experiments?|results?|discussion|conclusion|limitations?|future work|"
    r"acknowledg(e)?ments|references|bibliography)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _split_paper_sections(text: str) -> List[Dict[str, str]]:
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [{"section": "body", "text": text}]

    sections: List[Dict[str, str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group(0).strip()
        content = text[start:end].strip()
        sections.append({"section": title, "text": content})
    return sections


def _split_with_params(docs: List[Document], size: int, overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
    )
    return splitter.split_documents(docs)


def _split_paper_docs(docs: List[Document], size: int, overlap: int) -> List[Document]:
    derived: List[Document] = []
    for doc in docs:
        for section in _split_paper_sections(doc.page_content or ""):
            if not section["text"]:
                continue
            metadata = dict(doc.metadata)
            metadata["section"] = section["section"]
            derived.append(Document(page_content=section["text"], metadata=metadata))
    return _split_with_params(derived, size, overlap)


def split_documents(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    category_config: Dict[str, Dict[str, int]] | None = None,
) -> List[Document]:
    if not docs:
        return []

    category_config = category_config or {}
    default_cfg = {
        "size": chunk_size,
        "overlap": chunk_overlap,
    }

    grouped: Dict[str, List[Document]] = {}
    for doc in docs:
        category = doc.metadata.get("category", "default")
        grouped.setdefault(category, []).append(doc)

    results: List[Document] = []
    for category, items in grouped.items():
        cfg = category_config.get(category, default_cfg)
        size = int(cfg.get("size", default_cfg["size"]))
        overlap = int(cfg.get("overlap", default_cfg["overlap"]))
        if category == "paper":
            results.extend(_split_paper_docs(items, size, overlap))
        else:
            results.extend(_split_with_params(items, size, overlap))
    return results
