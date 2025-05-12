from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from PIL.Image import Image

from .firebase import storage
from .types import PartBlob, PartPagePdf


def _get_image_storage_path(
    part_blob: PartBlob, part_number: int, image_format: str = "png"
) -> Path:
    path_parts = list(part_blob.path.parts)

    # Get path till repositoryId
    group_index = path_parts.index("group")
    repertory_index = path_parts.index("repertory")
    path_parts[group_index] = "images"
    base_path = path_parts[: repertory_index + 2]

    # Add instrument folder
    base_path.append(part_blob.instrument_id)

    # Add filename
    group_id = path_parts[group_index + 1]
    repertory_id = path_parts[repertory_index + 1]
    file_name = f"{group_id}-{repertory_id}-{part_blob.path.name}-{part_number:05d}.{image_format}"
    base_path.append(file_name)

    return Path("/".join(base_path))


def _get_bytes_from_pil_image(image: Image) -> bytes:
    page_buffer = BytesIO()
    image.save(page_buffer, format="png")
    return page_buffer.getvalue()


def _upload_image_to_storage(path: Path, data: bytes) -> None:
    image_blob = storage.bucket().blob(path.as_posix())
    image_blob.upload_from_string(data, content_type="image/png")


def pdf2img(part_blob: PartBlob) -> list[PartPagePdf]:
    part_pages_images = []
    pages_images = pdfium.pdf_renderer.render_pdf_topil(
        part_blob.blob.download_as_bytes(), scale=4
    )
    for image in pages_images:
        page_render: Image = image[0]
        page_index = int(image[1])
        page_image_data = _get_bytes_from_pil_image(page_render)
        page_image_path = _get_image_storage_path(part_blob, page_index)
        _upload_image_to_storage(page_image_path, page_image_data)
        part_pages_images.append(PartPagePdf(part_blob.instrument_id, page_image_path))
    return part_pages_images
