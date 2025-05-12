from .firebase import db
from .types import PartPageImage, PartPagePdf, PartPdfWithSize


def classify_part_page_image(part_page_image: PartPageImage) -> None:
    ref = db.reference("partImagesClassification")
    return ref.push().set(part_page_image.to_classification_dict())


def classify_part_page_pdf(part_page_pdf: PartPagePdf) -> None:
    ref = db.reference("partPdfClassification")
    return ref.push().set(part_page_pdf.to_classification_str())


def classify_part_pdf_with_size(part_page_pdf: PartPdfWithSize) -> None:
    ref = db.reference("partPdfClassificationWithSize")
    return ref.push().set(part_page_pdf.to_classification_dict())
