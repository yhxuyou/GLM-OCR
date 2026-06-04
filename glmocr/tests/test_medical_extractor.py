from __future__ import annotations

import json

import pytest

from glmocr.medical_extractor import (
    BillRecord,
    FieldExtractionError,
    MedicalFieldExtractor,
)


class MockOCRClient:
    """模拟 OCR 客户端，按顺序返回预设响应"""

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def process(self, prompt, temperature=0.0):
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp


# ── BillRecord schema 测试 ──


class TestBillRecordSchema:
    """验证 BillRecord 包含所有必要字段"""

    def test_has_all_required_fields(self):
        """BillRecord 应包含 image_id/bill_no/hospital/issue_date/total_amount/self_pay/insurance_pay/items/raw_markdown 字段"""
        record = BillRecord(
            image_id="img-001",
            raw_markdown="# 票据",
        )
        expected_fields = {
            "image_id",
            "bill_no",
            "hospital",
            "issue_date",
            "total_amount",
            "self_pay",
            "insurance_pay",
            "items",
            "raw_markdown",
        }
        assert set(BillRecord.model_fields.keys()) == expected_fields


class TestBillRecordDefaults:
    """验证 BillRecord 各字段的默认值"""

    def test_optional_fields_default_none(self):
        """bill_no/hospital/issue_date/total_amount/self_pay/insurance_pay 默认为 None"""
        record = BillRecord(image_id="img-001", raw_markdown="# 票据")
        assert record.bill_no is None
        assert record.hospital is None
        assert record.issue_date is None
        assert record.total_amount is None
        assert record.self_pay is None
        assert record.insurance_pay is None

    def test_items_default_empty_list(self):
        """items 默认为空列表"""
        record = BillRecord(image_id="img-001", raw_markdown="# 票据")
        assert record.items == []

    def test_raw_markdown_required(self):
        """raw_markdown 为必填字段，缺少时应抛出验证错误"""
        with pytest.raises(Exception):
            BillRecord(image_id="img-001")


# ── extract 成功测试 ──


class TestExtractSuccess:
    """验证 extract 在正常情况下返回正确的 BillRecord"""

    def test_extract_returns_bill_record(self):
        """mock ocr_client.process 返回包含 ```json ... ``` 的文本，验证 extract 返回正确的 BillRecord"""
        bill_data = {
            "bill_no": "B12345",
            "hospital": "北京医院",
            "issue_date": "2025-01-15",
            "total_amount": 500.0,
            "self_pay": 200.0,
            "insurance_pay": 300.0,
            "items": [
                {
                    "name": "挂号费",
                    "quantity": 1,
                    "unit_price": 50.0,
                    "amount": 50.0,
                }
            ],
        }
        response_text = f"```json\n{json.dumps(bill_data, ensure_ascii=False)}\n```"
        mock_client = MockOCRClient([response_text])
        extractor = MedicalFieldExtractor(
            ocr_client=mock_client,
            prompt_template="提取: {markdown}",
        )

        result = extractor.extract(image_id="img-001", markdown="# 票据内容")

        assert isinstance(result, BillRecord)
        assert result.image_id == "img-001"
        assert result.bill_no == "B12345"
        assert result.hospital == "北京医院"
        assert result.issue_date == "2025-01-15"
        assert result.total_amount == 500.0
        assert result.self_pay == 200.0
        assert result.insurance_pay == 300.0
        assert len(result.items) == 1
        assert result.items[0].name == "挂号费"
        assert result.raw_markdown == "# 票据内容"


# ── extract JSON 损坏重试测试 ──


class TestExtractRetry:
    """验证 extract 在 JSON 损坏时能够重试并成功"""

    def test_retry_on_invalid_json(self):
        """第一次返回无效 JSON，第二次返回有效 JSON，验证重试成功"""
        valid_data = {
            "bill_no": "B999",
            "hospital": "上海医院",
            "issue_date": "2025-03-01",
            "total_amount": 100.0,
            "self_pay": 40.0,
            "insurance_pay": 60.0,
            "items": [],
        }
        # 第一次返回损坏的 JSON，第二次返回正常的 JSON
        responses = [
            "这不是JSON",
            f"```json\n{json.dumps(valid_data, ensure_ascii=False)}\n```",
        ]
        mock_client = MockOCRClient(responses)
        extractor = MedicalFieldExtractor(
            ocr_client=mock_client,
            prompt_template="提取: {markdown}",
            retry_count=1,
        )

        result = extractor.extract(image_id="img-002", markdown="# 票据")

        assert isinstance(result, BillRecord)
        assert result.bill_no == "B999"
        assert result.hospital == "上海医院"
        # 第一次调用（初始）+ 第二次调用（重试）= 2 次
        assert mock_client.call_count == 2


# ── extract 全部失败测试 ──


class TestExtractAllFail:
    """验证 extract 在所有尝试均失败时抛出 FieldExtractionError"""

    def test_raises_field_extraction_error(self):
        """mock ocr_client.process 始终返回无效 JSON，验证抛出 FieldExtractionError"""
        mock_client = MockOCRClient(["坏数据", "还是坏数据"])
        extractor = MedicalFieldExtractor(
            ocr_client=mock_client,
            prompt_template="提取: {markdown}",
            retry_count=1,
        )

        with pytest.raises(FieldExtractionError):
            extractor.extract(image_id="img-003", markdown="# 票据")


# ── _extract_json_from_text 测试 ──


class TestExtractJsonFromText:
    """验证 _extract_json_from_text 能从不同格式的文本中提取 JSON"""

    def test_extract_from_json_code_block(self):
        """从 ```json ... ``` 代码块中提取 JSON"""
        text = '一些说明\n```json\n{"key": "value"}\n```\n后续文本'
        result = MedicalFieldExtractor._extract_json_from_text(text)
        assert json.loads(result) == {"key": "value"}

    def test_extract_from_plain_code_block(self):
        """从无语言标记的 ``` ... ``` 代码块中提取 JSON"""
        text = '一些说明\n```\n{"key": "value"}\n```\n后续文本'
        result = MedicalFieldExtractor._extract_json_from_text(text)
        assert json.loads(result) == {"key": "value"}

    def test_extract_from_plain_text(self):
        """从纯文本（无代码块标记）中提取 JSON"""
        text = '{"key": "value"}'
        result = MedicalFieldExtractor._extract_json_from_text(text)
        assert json.loads(result) == {"key": "value"}
