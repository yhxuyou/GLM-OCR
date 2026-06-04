from __future__ import annotations

import pytest

from glmocr.medical_aggregator import (
    AggregationResult,
    CaseInfo,
    MedicalAggregator,
    MedicalRecordInput,
)


# ── Mock Bill 对象 ──────────────────────────────────────────────────


class MockBill:
    def __init__(
        self,
        image_id,
        hospital="",
        issue_date="",
        total_amount=None,
        bill_no=None,
        items=None,
        raw_markdown="",
    ):
        self.image_id = image_id
        self.hospital = hospital
        self.issue_date = issue_date
        self.total_amount = total_amount
        self.bill_no = bill_no
        self.items = items or []
        self.raw_markdown = raw_markdown


# ── 测试用例 ────────────────────────────────────────────────────────


class TestMatchByHospitalAndDate:
    """票据 hospital+issue_date 与病历 hospital_name+startDate/endDate 匹配成功"""

    def test_match_success(self):
        agg = MedicalAggregator(date_window_days=7)
        bill = MockBill(
            image_id="img_001",
            hospital="北京协和医院",
            issue_date="2024-03-15",
            total_amount=100.0,
            raw_markdown="bill1",
        )
        records = [
            MedicalRecordInput(
                medical_id="med_01",
                hospital_name="协和医院",
                startDate="2024-03-10",
                endDate="2024-03-20",
            )
        ]
        result = agg.aggregate([bill], records)
        assert len(result.case_info) == 1
        assert result.case_info[0].case_id == "med_01"
        assert len(result.case_info[0].bills) == 1
        assert result.discarded_image == []


class TestMatchHospitalOnly:
    """只有医院名称匹配（日期为空），也能匹配成功"""

    def test_match_hospital_no_date(self):
        agg = MedicalAggregator()
        bill = MockBill(
            image_id="img_002",
            hospital="上海瑞金医院",
            issue_date="",  # 票据日期为空
            total_amount=200.0,
            raw_markdown="bill2",
        )
        records = [
            MedicalRecordInput(
                medical_id="med_02",
                hospital_name="瑞金医院",
                startDate="",  # 病历日期也为空
                endDate="",
            )
        ]
        result = agg.aggregate([bill], records)
        assert len(result.case_info) == 1
        assert result.case_info[0].case_id == "med_02"


class TestNoMatch:
    """医院名不匹配，返回 None（票据进入 discarded_image）"""

    def test_hospital_mismatch(self):
        agg = MedicalAggregator()
        bill = MockBill(
            image_id="img_003",
            hospital="北京大学第一医院",
            issue_date="2024-03-15",
            total_amount=300.0,
            raw_markdown="bill3",
        )
        records = [
            MedicalRecordInput(
                medical_id="med_03",
                hospital_name="复旦大学附属中山医院",
                startDate="2024-03-10",
                endDate="2024-03-20",
            )
        ]
        result = agg.aggregate([bill], records)
        assert len(result.case_info) == 0
        assert "img_003" in result.discarded_image


class TestBillTotalSum:
    """同 case 下多张票据，bill_total = sum(total_amount)"""

    def test_sum_multiple_bills(self):
        agg = MedicalAggregator()
        bill1 = MockBill(
            image_id="img_010",
            hospital="协和医院",
            issue_date="2024-03-15",
            total_amount=100.0,
            raw_markdown="bill_a",
        )
        bill2 = MockBill(
            image_id="img_011",
            hospital="协和医院",
            issue_date="2024-03-16",
            total_amount=250.5,
            raw_markdown="bill_b",
        )
        records = [
            MedicalRecordInput(
                medical_id="med_10",
                hospital_name="协和医院",
                startDate="2024-03-10",
                endDate="2024-03-20",
            )
        ]
        result = agg.aggregate([bill1, bill2], records)
        assert len(result.case_info) == 1
        assert result.case_info[0].bill_total == pytest.approx(350.5)


class TestNoMedicalRecords:
    """病历列表为空，所有票据进 discarded_image，case_info 为 []"""

    def test_empty_medical_records(self):
        agg = MedicalAggregator()
        bill1 = MockBill(
            image_id="img_020",
            hospital="协和医院",
            total_amount=100.0,
            raw_markdown="bill_x",
        )
        bill2 = MockBill(
            image_id="img_021",
            hospital="瑞金医院",
            total_amount=200.0,
            raw_markdown="bill_y",
        )
        result = agg.aggregate([bill1, bill2], [])
        assert result.case_info == []
        assert "img_020" in result.discarded_image
        assert "img_021" in result.discarded_image


class TestDedupDuplicateMarkdown:
    """raw_markdown hash 一致的图只保留首张，后续进 discarded_image"""

    def test_dedup(self):
        agg = MedicalAggregator()
        same_markdown = "这是一段相同的markdown内容"
        bill1 = MockBill(
            image_id="img_030",
            hospital="协和医院",
            issue_date="2024-03-15",
            total_amount=100.0,
            raw_markdown=same_markdown,
        )
        bill2 = MockBill(
            image_id="img_031",
            hospital="协和医院",
            issue_date="2024-03-15",
            total_amount=100.0,
            raw_markdown=same_markdown,  # 重复
        )
        records = [
            MedicalRecordInput(
                medical_id="med_30",
                hospital_name="协和医院",
                startDate="2024-03-10",
                endDate="2024-03-20",
            )
        ]
        result = agg.aggregate([bill1, bill2], records)
        # 只有第一张保留在 case 中
        assert len(result.case_info[0].bills) == 1
        assert result.case_info[0].bills[0].image_id == "img_030"
        # 第二张进入 discarded
        assert "img_031" in result.discarded_image


class TestUnmatchedBillDiscarded:
    """未匹配上的票据 image_id 进入 discarded_image，case_info 不出现"""

    def test_unmatched_discarded(self):
        agg = MedicalAggregator()
        matched_bill = MockBill(
            image_id="img_040",
            hospital="协和医院",
            issue_date="2024-03-15",
            total_amount=100.0,
            raw_markdown="matched",
        )
        unmatched_bill = MockBill(
            image_id="img_041",
            hospital="不匹配的医院",
            issue_date="2024-03-15",
            total_amount=200.0,
            raw_markdown="unmatched",
        )
        records = [
            MedicalRecordInput(
                medical_id="med_40",
                hospital_name="协和医院",
                startDate="2024-03-10",
                endDate="2024-03-20",
            )
        ]
        result = agg.aggregate([matched_bill, unmatched_bill], records)
        # 匹配成功的票据在 case_info 中
        assert len(result.case_info) == 1
        bill_ids = [b.image_id for b in result.case_info[0].bills]
        assert "img_040" in bill_ids
        # 未匹配票据不在 case_info 中
        assert "img_041" not in bill_ids
        # 未匹配票据在 discarded_image 中
        assert "img_041" in result.discarded_image


class TestDiscardedImageSorted:
    """discarded_image 按 seq 升序排列"""

    def test_sorted_by_seq(self):
        agg = MedicalAggregator()
        # 病历为空，所有票据都进 discarded
        bills = [
            MockBill(image_id="img_005", raw_markdown="c"),
            MockBill(image_id="img_002", raw_markdown="a"),
            MockBill(image_id="img_010", raw_markdown="b"),
            MockBill(image_id="img_001", raw_markdown="d"),
        ]
        result = agg.aggregate(bills, [])
        # 按 image_id 中的数字升序排列：1, 2, 5, 10
        assert result.discarded_image == [
            "img_001",
            "img_002",
            "img_005",
            "img_010",
        ]
