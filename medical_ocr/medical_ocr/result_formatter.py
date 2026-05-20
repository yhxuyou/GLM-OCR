"""Medical Result Formatter - Specialized post-processing for medical documents.

Inherits from glmocr's ResultFormatter and adds medical-specific post-processing.
"""

import json
import re
from copy import deepcopy
from typing import List, Dict, Tuple, Any

from glmocr.postprocess import ResultFormatter
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class MedicalResultFormatter(ResultFormatter):
    """Medical-specific result formatter.

    Adds medical document specific post-processing:
    - Medical terminology normalization
    - Medical unit standardization
    - Clean up common medical OCR artifacts
    """

    def __init__(self, config):
        super().__init__(config)
        
        # Medical-specific configuration
        self.enable_medical_normalization = getattr(
            config, "enable_medical_normalization", True
        )
        self.enable_unit_conversion = getattr(
            config, "enable_unit_conversion", True
        )

    # =========================================================================
    # Medical-specific post-processing
    # =========================================================================

    def _normalize_medical_terminology(self, content: str) -> str:
        """Normalize medical terminology in content."""
        if not self.enable_medical_normalization:
            return content

        # Common medical abbreviations and terminology
        medical_patterns = [
            # Temperature
            (r"\b(\d+(?:\.\d+)?)\s*(°?C)\b", r"\1°C"),
            (r"\b(\d+(?:\.\d+)?)\s*(°?F)\b", r"\1°F"),
            
            # Blood pressure
            (r"\b(\d{2,3})/(\d{2,3})\s*(mmHg)?\b", r"\1/\2 mmHg"),
            
            # Weight
            (r"\b(\d+(?:\.\d+)?)\s*(kg)\b", r"\1 kg"),
            (r"\b(\d+(?:\.\d+)?)\s*(g)\b", r"\1 g"),
            
            # Height
            (r"\b(\d+(?:\.\d+)?)\s*(cm)\b", r"\1 cm"),
            (r"\b(\d+(?:\.\d+)?)\s*(m)\b", r"\1 m"),
            
            # Medical abbreviations
            (r"\bBP\b", "Blood Pressure"),
            (r"\bHR\b", "Heart Rate"),
            (r"\bRR\b", "Respiratory Rate"),
            (r"\bSPO2\b", "SpO₂"),
            (r"\bBMI\b", "BMI"),
            
            # Date formats
            (r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", r"\1-\2-\3"),
        ]

        for pattern, replacement in medical_patterns:
            content = re.sub(pattern, replacement, content)

        return content

    def _clean_medical_ocr_artifacts(self, content: str) -> str:
        """Clean common OCR artifacts in medical documents."""
        # Fix common OCR errors in medical context
        fixes = [
            # Numbers and letters
            ("l", "1"),  # lowercase L to 1
            ("O", "0"),  # uppercase O to 0
            ("o", "0"),  # lowercase o to 0
            
            # Medical symbols
            ("\\", "/"),  # Backslash to forward slash
            ("~", "-"),   # Tilde to hyphen
            
            # Remove extra whitespace around numbers
            (r"\s+(\d+)\s+", r" \1 "),
            
            # Normalize spaces around slashes
            (r"\s*/\s*", "/"),
        ]

        for old, new in fixes:
            if isinstance(old, str):
                content = content.replace(old, new)
            else:
                content = re.sub(old, new, content)

        return content

    def _format_medical_content(self, content: str, label: str) -> str:
        """Apply medical-specific formatting."""
        if content is None:
            return content

        content = str(content)

        # Clean OCR artifacts
        content = self._clean_medical_ocr_artifacts(content)

        # Normalize medical terminology
        content = self._normalize_medical_terminology(content)

        # Special handling for tables
        if label == "table":
            content = self._format_medical_table(content)

        return content

    def _format_medical_table(self, content: str) -> str:
        """Format medical tables for better readability."""
        if not content.startswith("<table"):
            return content

        # Add spacing around table cells
        content = re.sub(r"<td>", "<td> ", content)
        content = re.sub(r"</td>", " </td>", content)

        # Ensure proper newline handling
        content = content.replace("</tr>", "</tr>\n")

        return content

    # =========================================================================
    # Override parent methods
    # =========================================================================

    def _clean_content(self, content: str) -> str:
        """Clean content with medical-specific enhancements."""
        content = super()._clean_content(content)
        
        # Add medical-specific cleaning
        content = self._clean_medical_ocr_artifacts(content)
        
        return content

    def _format_content(self, content: Any, label: str, native_label: str) -> str:
        """Format content with medical-specific enhancements."""
        # First apply parent formatting
        content = super()._format_content(content, label, native_label)
        
        # Then apply medical-specific formatting
        content = self._format_medical_content(content, label)
        
        return content

    def format_ocr_result(self, content: str, page_idx: int = 0) -> Tuple[str, str]:
        """Format OCR result with medical enhancements."""
        # Clean content first
        content = self._clean_medical_ocr_artifacts(content)
        content = self._normalize_medical_terminology(content)
        
        # Call parent method
        return super().format_ocr_result(content, page_idx)

    def process(
        self,
        grouped_results: List[List[Dict]],
        cropped_images: Dict[tuple, Any] | None = None,
        image_prefix: str = "cropped",
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Process grouped results with medical enhancements."""
        # First apply medical-specific preprocessing to results
        preprocessed_results = []
        for page_results in grouped_results:
            preprocessed_page = []
            for item in page_results:
                item_copy = deepcopy(item)
                
                # Apply medical formatting to content
                if "content" in item_copy and item_copy["content"]:
                    label = item_copy.get("label", "text")
                    item_copy["content"] = self._format_medical_content(
                        item_copy["content"], label
                    )
                
                preprocessed_page.append(item_copy)
            preprocessed_results.append(preprocessed_page)

        # Call parent process with preprocessed results
        json_str, markdown_str, image_files = super().process(
            preprocessed_results,
            cropped_images=cropped_images,
            image_prefix=image_prefix,
        )

        # Apply additional medical post-processing to markdown
        markdown_str = self._post_process_markdown(markdown_str)

        return json_str, markdown_str, image_files

    def _post_process_markdown(self, markdown: str) -> str:
        """Apply final medical-specific markdown processing."""
        # Add section headers for medical documents
        lines = markdown.split("\n")
        processed_lines = []
        
        for i, line in enumerate(lines):
            # Add emphasis to medical terms
            line = self._emphasize_medical_terms(line)
            processed_lines.append(line)
        
        return "\n".join(processed_lines)

    def _emphasize_medical_terms(self, text: str) -> str:
        """Add emphasis to common medical terms."""
        medical_terms = [
            "Blood Pressure", "Heart Rate", "Respiratory Rate",
            "SpO₂", "BMI", "Temperature",
            "mmHg", "kg", "cm", "°C", "°F",
        ]
        
        for term in medical_terms:
            pattern = re.escape(term)
            text = re.sub(rf"\b({pattern})\b", r"**\1**", text)
        
        return text
