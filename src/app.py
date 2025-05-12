from fastapi import FastAPI, Security

from . import __project_name__
from .auth import JWTBearer
from .lib.classification import (
    classify_part_page_image,
    classify_part_page_pdf,
    classify_part_pdf_with_size,
)
from .lib.pdf2img import pdf2img
from .lib.split_pdf import split_pdf
from .lib.tag_file_size import tag_file_size
from .models import Input, InputWithInstrumentId

app = FastAPI(title=__project_name__, dependencies=[Security(JWTBearer())])


@app.post("/pdf2img")
async def pdf2img_endpoint(input: Input) -> None:
    part_page_images = pdf2img(input.part_blob)
    for part_page_image in part_page_images:
        classify_part_page_image(part_page_image)


@app.post("/split_pdf")
async def split_pdf_endpoint(input: Input) -> None:
    part_page_pdfs = split_pdf(input.part_blob)
    for part_page_pdf in part_page_pdfs:
        classify_part_page_pdf(part_page_pdf)


@app.post("/tag_file_size")
async def tag_file_size_endpoint(input: InputWithInstrumentId) -> None:
    part_pdf = tag_file_size(input.part_blob)
    classify_part_pdf_with_size(part_pdf)
