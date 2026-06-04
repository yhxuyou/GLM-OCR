from __future__ import annotations

import json
import re
from typing import List, Optional

from pydantic import BaseModel, Field


class BillItem(BaseModel):
    """医疗票据费用条目"""

    name: str
    quantity: float
    unit_price: float
    amount: float


class BillRecord(BaseModel):
    """医疗票据完整记录"""

    image_id: str
    bill_no: Optional[str] = None
    hospital: Optional[str] = None
    issue_date: Optional[str] = None
    total_amount: Optional[float] = None
    self_pay: Optional[float] = None
    insurance_pay: Optional[float] = None
    items: List[BillItem] = []
    raw_markdown: str


class FieldExtractionError(Exception):
    """字段抽取失败异常"""
    pass


class MedicalFieldExtractor:
    """医疗票据字段抽取器，基于 LLM 从 OCR 识别结果中提取结构化字段"""

    def __init__(
        self,
        ocr_client,
        prompt_template: str,
        retry_count: int = 1,
        temperature: float = 0.0,
        retry_temperature: float = 0.3,
    ):
        self.ocr_client = ocr_client
        self.prompt_template = prompt_template
        self.retry_count = retry_count
        self.temperature = temperature
        self.retry_temperature = retry_temperature

    def extract(self, image_id: str, markdown: str) -> BillRecord:
        """主入口方法：从 markdown 文本中提取医疗票据字段

        Args:
            image_id: 图片标识
            markdown: OCR 识别得到的 markdown 文本

        Returns:
            BillRecord: 解析后的医疗票据记录

        Raises:
            FieldExtractionError: 所有尝试均解析失败时抛出
        """
        prompt = self.prompt_template.format(markdown=markdown)

        # 首次尝试使用 temperature
        last_error: Exception | None = None
        try:
            response_text = self.ocr_client.process(prompt, temperature=self.temperature)
            json_str = self._extract_json_from_text(response_text)
            return self._parse_bill_record(json_str, image_id, markdown)
        except (json.JSONDecodeError, ValueError, FieldExtractionError) as exc:
            last_error = exc

        # 重试 retry_count 次，使用 retry_temperature
        for _ in range(self.retry_count):
            try:
                response_text = self.ocr_client.process(
                    prompt, temperature=self.retry_temperature
                )
                json_str = self._extract_json_from_text(response_text)
                return self._parse_bill_record(json_str, image_id, markdown)
            except (json.JSONDecodeError, ValueError, FieldExtractionError) as exc:
                last_error = exc

        raise FieldExtractionError(
            f"字段抽取失败，已重试 {self.retry_count} 次，最后错误: {last_error}"
        )

    def _parse_bill_record(
        self, json_str: str, image_id: str, markdown: str
    ) -> BillRecord:
        """将 JSON 字符串解析为 BillRecord，并填充 image_id 和 raw_markdown"""
        data = json.loads(json_str)
        data["image_id"] = image_id
        data["raw_markdown"] = markdown
        return BillRecord.model_validate(data)

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """从 LLM 输出文本中提取 JSON 字符串

        优先尝试提取 ```json ... ``` 代码块中的内容，
        若无代码块则直接使用整段文本作为 JSON 字符串。

        Args:
            text: LLM 返回的文本

        Returns:
            提取出的 JSON 字符串
        """
        # 尝试匹配 ```json ... ``` 代码块
        match = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 尝试匹配 ``` ... ``` 代码块（无语言标记）
        match = re.search(r"```\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 直接返回原文
        return text.strip()
