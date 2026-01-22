from dataclasses import dataclass
from pathlib import Path
import fitz  # PyMuPDF
from streamlit.runtime.uploaded_file_manager import UploadedFile
from config import Config
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import io
import hashlib
from typing import Optional, List, Dict
from table_intelligence import TableExtractor, StructuredTable
import json
import re

TEXT_FILE_EXTENSION = ".txt"
MD_FILE_EXTENSION = '.md'
PDF_FILE_EXTENSION = ".pdf"

# global variables for lazy loading
_got_model = None
_got_processor = None

def get_got_ocr_model():
    """
    Lazy-load GOT-OCR2 model.
    """
    global _got_model, _got_processor
    
    if _got_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # load model normally with float16 for GPU efficiency
        _got_model = AutoModelForImageTextToText.from_pretrained(
            "stepfun-ai/GOT-OCR-2.0-hf",
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        
        _got_processor = AutoProcessor.from_pretrained(
            "stepfun-ai/GOT-OCR-2.0-hf",
            use_fast=True
        )
        
        # set to eval mode
        _got_model.eval()        
        if device == "cuda":
            print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
    return _got_model, _got_processor

def run_got_ocr_on_image(image_bytes: bytes, format_output: bool = True) -> str:
    """
    Run GOT-OCR2 on image bytes and return extracted text.
    """
    try:
        model, processor = get_got_ocr_model()
        
        # convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # validate image
        if image.size[0] < 10 or image.size[1] < 10:
            return ""  # skip tiny images
        
        # process image with appropriate settings
        inputs = processor(
            image,
            return_tensors="pt",
            format=format_output  # enable for lateX, tables, formulas
        )
        
        # move to GPU if available
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # generate with memory-efficient settings
        with torch.no_grad(): 
            generate_ids = model.generate(
                **inputs,
                do_sample=False,
                tokenizer=processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=2048,
                num_beams=1,  # greedy decoding
            )
        
        # decode output
        output_text = processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # clear cuda cache after each inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_text.strip()
        
    except Exception as e:
        import traceback
        print(f"GOT-OCR2 failed: {e}")
        print(traceback.format_exc())
        return ""

@dataclass
class ContentBlock:
    """Represents a block of content from PDF"""
    content: str
    content_type: str  # 'text', 'table', 'figure'
    page_num: int
    bbox: Optional[tuple] = None
    table_data: Optional[StructuredTable] = None

def extract_pdf_content_with_structure(data: bytes, use_got_ocr: bool = True) -> List[ContentBlock]:
    """
    Extract content from PDF with structure detection.
    """
    with fitz.open(stream=data, filetype="pdf") as doc:
        all_blocks: List[ContentBlock] = []
        table_extractor = TableExtractor()

        caption_re = re.compile(r"^\s*(table)\s+\d+(?:\.\d+)*\s*[:.-]", re.IGNORECASE)

        for page_num, page in enumerate(doc):
            print(f"Processing page {page_num + 1}/{len(doc)}...")

            # one pass: get layout blocks in reading order
            layout_blocks = page.get_text("dict", sort=True).get("blocks", [])

            page_blocks: List[ContentBlock] = []

            # native tables 
            found_tables = page.find_tables()
            table_bboxes: List[tuple] = []

            for table in found_tables:
                header = table.header.names
                if not header:
                    header = [f"Col {i+1}" for i in range(table.col_count)]
                else:
                    # convert None values to strings
                    header = [str(h) if h is not None else f"Col {i+1}"
                              for i, h in enumerate(header)]

                rows = []
                for row_data in table.extract():
                    if row_data == table.header.names:
                        continue
                    cleaned_row = [str(cell) if cell is not None else "" for cell in row_data]
                    if len(cleaned_row) == len(header):
                        rows.append(dict(zip(header, cleaned_row)))

                # markdown representation
                markdown_lines = [
                    "| " + " | ".join(header) + " |",
                    "| " + " | ".join(["---"] * len(header)) + " |",
                ]
                for row in rows:
                    row_values = [str(row.get(h, "")) for h in header]
                    markdown_lines.append("| " + " | ".join(row_values) + " |")
                raw_markdown = "\n".join(markdown_lines)

                structured_table = StructuredTable(
                    headers=header,
                    rows=rows,
                    raw_markdown=raw_markdown,
                    num_rows=len(rows),
                    num_cols=len(header),
                )

                tbbox = tuple(table.bbox)
                table_bboxes.append(tbbox)

                page_blocks.append(
                    ContentBlock(
                        content=raw_markdown,
                        content_type="table",
                        page_num=page_num + 1,
                        bbox=tbbox,
                        table_data=structured_table,
                    )
                )

            def _overlaps_table(bbox: tuple) -> bool:
                if not bbox or bbox == (0, 0, 0, 0):
                    return False
                b = fitz.Rect(bbox)
                for t_bbox in table_bboxes:
                    if b.intersects(fitz.Rect(t_bbox)):
                        return True
                return False

            # Text + image blocks (OCR for image blocks) 
            for block in layout_blocks:
                bbox = tuple(block.get("bbox", (0, 0, 0, 0)))

                if _overlaps_table(bbox):
                    continue

                if block.get("type") == 0:
                    text_content = []
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_content.append(span.get("text", ""))
                    content = "\n".join([t for t in text_content if t is not None])

                    if content.strip():
                        page_blocks.append(
                            ContentBlock(
                                content=content,
                                content_type="text",
                                page_num=page_num + 1,
                                bbox=bbox,
                            )
                        )

                elif block.get("type") == 1 and use_got_ocr:
                    try:
                        if bbox:
                            mat = fitz.Matrix(2, 2)
                            pix = page.get_pixmap(matrix=mat, clip=bbox)
                            image_bytes = pix.tobytes("png")

                            extracted_text = run_got_ocr_on_image(image_bytes, format_output=True)

                            if extracted_text:
                                table_result = table_extractor.extract_table_from_ocr(extracted_text)
                                if table_result:
                                    structured_table, original_text = table_result
                                    page_blocks.append(
                                        ContentBlock(
                                            content=original_text,
                                            content_type="table",
                                            page_num=page_num + 1,
                                            bbox=bbox,
                                            table_data=structured_table,
                                        )
                                    )
                                else:
                                    page_blocks.append(
                                        ContentBlock(
                                            content=extracted_text,
                                            content_type="figure",
                                            page_num=page_num + 1,
                                            bbox=bbox,
                                        )
                                    )
                    except Exception as e:
                        print(f"Failed to process image on page {page_num + 1}: {e}")

            # attach captions to nearest table 
            caption_block_ids_to_drop = set()
            table_blocks = [b for b in page_blocks if b.content_type == "table" and b.bbox]
            text_blocks = [b for b in page_blocks if b.content_type == "text" and b.bbox]

            for tb in table_blocks:
                trect = fitz.Rect(tb.bbox)
                best = None  # (distance, text_block)

                for tx in text_blocks:
                    if id(tx) in caption_block_ids_to_drop:
                        continue
                    candidate = tx.content.strip()
                    if not candidate or not caption_re.match(candidate):
                        continue

                    txrect = fitz.Rect(tx.bbox)

                    # must be reasonably near vertically
                    if txrect.y0 >= trect.y1:
                        dist = txrect.y0 - trect.y1  # below
                    else:
                        dist = trect.y0 - txrect.y1  # above

                    if dist < 0 or dist > 120:
                        continue

                    # require some horizontal overlap (same column-ish)
                    x_overlap = max(0.0, min(trect.x1, txrect.x1) - max(trect.x0, txrect.x0))
                    if x_overlap < (min(trect.width, txrect.width) * 0.2):
                        continue

                    if best is None or dist < best[0]:
                        best = (dist, tx)

                if best:
                    _, cap_block = best
                    caption_text = cap_block.content.strip()
                    if caption_text and caption_text not in tb.content:
                        tb.content = f"{caption_text}\n{tb.content}"
                    caption_block_ids_to_drop.add(id(cap_block))

            page_blocks = [b for b in page_blocks if id(b) not in caption_block_ids_to_drop]

            # keep reading order 
            def _sort_key(b: ContentBlock):
                if b.bbox:
                    x0, y0, x1, y1 = b.bbox
                    return (y0, x0)
                return (float("inf"), float("inf"))

            page_blocks.sort(key=_sort_key)
            all_blocks.extend(page_blocks)

        return all_blocks


def format_content_blocks_as_text(content_blocks: List[ContentBlock]) -> str:
    """
    Convert structured content blocks back to text format for storage.

    Uses markers compatible with data_ingestor.py:
      [TABLE: ...] ... [/TABLE]
      [FIGURE] ... [/FIGURE]
    """
    output: List[str] = []

    for block in content_blocks:
        output.append(f"\n--- PAGE {block.page_num} ---\n")

        if block.content_type == "table":
            output.append("[TABLE:]")

            # caption + markdown table (if caption was attached)
            if block.content and block.content.strip():
                output.append(block.content.strip())

            # extra BM25-friendly searchable table text
            if block.table_data:
                output.append("")
                output.append(block.table_data.to_searchable_text())

            output.append("[/TABLE]")
            output.append("")
            continue

        if block.content_type == "figure":
            output.append("[FIGURE]")
            output.append((block.content or "").strip())
            output.append("[/FIGURE]")
            output.append("")
            continue

        output.append((block.content or "").strip())
        output.append("")

    return "\n".join(output)


def extract_pdf_content(data: bytes, use_got_ocr: bool = True) -> str:
    """
    Extract content from PDF with enhanced table detection.
    This is the main function used by the system.
    """
    # extract with structure
    content_blocks = extract_pdf_content_with_structure(data, use_got_ocr)
    
    # convert to text format
    return format_content_blocks_as_text(content_blocks)

@dataclass 
class File:
    name: str
    content: str
    content_blocks: Optional[List[ContentBlock]] = None  # store structured version


def load_uploaded_file(uploaded_file: UploadedFile) -> File:
    """
    Load and process uploaded file with persistent disk caching.
    Enhanced with table intelligence.
    """
    file_extension = Path(uploaded_file.name).suffix
    
    if file_extension not in Config.ALLOWED_FILE_EXTENSIONS:
        raise ValueError(
            f"Invalid file extension: {file_extension} for file {uploaded_file.name}"
        )
    
    # get raw bytes to generate a unique hash
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    # define cache directory and file path
    cache_dir = Config.Path.DATA_DIR / "ocr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{file_hash}.txt"
    cache_blocks_path = cache_dir / f"{file_hash}_blocks.json"

    # load from disk if file is processed already
    if cache_path.exists():
        print(f"Cache hit: Loading processed text for {uploaded_file.name}")
        with open(cache_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # try to load structured blocks if available
        content_blocks = None
        if cache_blocks_path.exists():
            try:
                with open(cache_blocks_path, "r", encoding="utf-8") as f:
                    blocks_data = json.load(f)
                    # reconstruct ContentBlock objects
                    content_blocks = []
                    for block_dict in blocks_data:
                        table_data = None
                        if block_dict.get('table_data'):
                            td = block_dict['table_data']
                            table_data = StructuredTable(
                                headers=td['headers'],
                                rows=td['rows'],
                                raw_markdown=td['raw_markdown'],
                                num_rows=td['num_rows'],
                                num_cols=td['num_cols']
                            )
                        
                        content_blocks.append(ContentBlock(
                            content=block_dict['content'],
                            content_type=block_dict['content_type'],
                            page_num=block_dict['page_num'],
                            bbox=tuple(block_dict['bbox']) if block_dict.get('bbox') else None,
                            table_data=table_data
                        ))
            except Exception as e:
                print(f"Warning: Could not load structured blocks: {e}")
        
        return File(name=uploaded_file.name, content=content, content_blocks=content_blocks)
    
    # if not in cache, run the extraction
    print(f"Running extraction for {uploaded_file.name}")
    
    if file_extension == PDF_FILE_EXTENSION:
        # Extract with structure
        content_blocks = extract_pdf_content_with_structure(file_bytes, use_got_ocr=True)
        
        # Convert to text
        content = format_content_blocks_as_text(content_blocks)
        
        # Save structured blocks to cache
        try:
            blocks_data = []
            for block in content_blocks:
                block_dict = {
                    'content': block.content,
                    'content_type': block.content_type,
                    'page_num': block.page_num,
                    'bbox': list(block.bbox) if block.bbox else None,
                    'table_data': block.table_data.to_dict() if block.table_data else None
                }
                blocks_data.append(block_dict)
            
            with open(cache_blocks_path, "w", encoding="utf-8") as f:
                json.dump(blocks_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not cache structured blocks: {e}")
        
    else:
        content = file_bytes.decode("utf-8")
        content_blocks = None
    
    # save the result to avoid running OCR again
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    return File(name=uploaded_file.name, content=content, content_blocks=content_blocks)


# cleanup function to free GPU memory when done
def cleanup_got_model():
    """call this when you're done processing to free GPU memory."""
    global _got_model, _got_processor
    
    if _got_model is not None:
        del _got_model
        del _got_processor
        _got_model = None
        _got_processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GOT-OCR2 model unloaded, GPU memory freed")