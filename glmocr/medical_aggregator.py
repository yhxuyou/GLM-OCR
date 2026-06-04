from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Optional

from pydantic import BaseModel


# ── Pydantic 模型定义 ──────────────────────────────────────────────


class MedicalRecordInput(BaseModel):
    """病历输入"""

    medical_id: str
    hospital_name: str = ""
    outpatientDate: str = ""
    startDate: str = ""
    endDate: str = ""


class PassThroughData(BaseModel):
    """透传数据，包含病历记录列表"""

    medical_records: List[MedicalRecordInput] = []


class CaseBill(BaseModel):
    """归并到某个 case 下的一张票据"""

    image_id: str
    bill_no: Optional[str] = None
    total_amount: Optional[float] = None
    items: List[dict] = []


class CaseInfo(BaseModel):
    """一个 case 的归并结果"""

    case_id: str
    hospital_name: str = ""
    bill_total: float = 0.0
    bills: List[CaseBill] = []


class AggregationResult(BaseModel):
    """归并结果：被丢弃的 image_id 列表 + case 维度归并信息"""

    discarded_image: List[str] = []
    case_info: List[CaseInfo] = []


# ── 归并器实现 ─────────────────────────────────────────────────────


class MedicalAggregator:
    """病历-票据匹配 + case 维度归并"""

    def __init__(self, date_window_days: int = 7) -> None:
        self.date_window_days = date_window_days

    # ── 公开接口 ───────────────────────────────────────────────────

    def aggregate(
        self,
        bills: list,
        medical_records: List[MedicalRecordInput],
    ) -> AggregationResult:
        """
        主入口：对票据列表进行去重、匹配、归并。

        参数:
            bills: 票据对象列表（duck typing，需具有 image_id / hospital /
                   issue_date / total_amount / bill_no / items / raw_markdown 属性）
            medical_records: 病历记录列表

        返回:
            AggregationResult
        """
        discarded_image: List[str] = []

        # ── a. 按 raw_markdown SHA1 去重 ──────────────────────────
        seen_hashes: dict[str, str] = {}  # hash -> image_id
        unique_bills: list = []

        for bill in bills:
            text = getattr(bill, "raw_markdown", "") or ""
            h = self._compute_hash(text)
            if h in seen_hashes:
                # 重复票据，加入 discarded
                discarded_image.append(bill.image_id)
            else:
                seen_hashes[h] = bill.image_id
                unique_bills.append(bill)

        # ── f. medical_records 为空时，所有 bill 进 discarded ─────
        if not medical_records:
            for bill in unique_bills:
                discarded_image.append(bill.image_id)
            # 按 image_id 中的 seq 数字升序排列
            discarded_image.sort(key=self._extract_seq)
            return AggregationResult(discarded_image=discarded_image, case_info=[])

        # ── b. 对去重后的每张 bill 进行匹配 ───────────────────────
        # case_id -> CaseInfo 的中间映射
        case_map: dict[str, CaseInfo] = {}

        for bill in unique_bills:
            medical_id = self._match_bill_to_medical(
                bill, medical_records, self.date_window_days
            )
            if medical_id is None:
                # ── e. 匹配失败，加入 discarded ───────────────────
                discarded_image.append(bill.image_id)
            else:
                # ── d. 匹配成功，归入对应 case ────────────────────
                if medical_id not in case_map:
                    # 找到对应病历记录以获取医院名称
                    mr = next(
                        (m for m in medical_records if m.medical_id == medical_id),
                        None,
                    )
                    hospital_name = mr.hospital_name if mr else ""
                    case_map[medical_id] = CaseInfo(
                        case_id=medical_id,
                        hospital_name=hospital_name,
                    )
                case_bill = CaseBill(
                    image_id=bill.image_id,
                    bill_no=getattr(bill, "bill_no", None),
                    total_amount=getattr(bill, "total_amount", None),
                    items=getattr(bill, "items", []) or [],
                )
                case_map[medical_id].bills.append(case_bill)

        # ── g. 计算 bill_total ────────────────────────────────────
        for case_info in case_map.values():
            case_info.bill_total = sum(b.total_amount or 0 for b in case_info.bills)

        # ── h. discarded_image 按 image_id 中的 seq 数字升序 ──────
        discarded_image.sort(key=self._extract_seq)

        return AggregationResult(
            discarded_image=discarded_image,
            case_info=list(case_map.values()),
        )

    # ── 私有方法 ───────────────────────────────────────────────────

    @staticmethod
    def _match_bill_to_medical(
        bill,
        medical_records: List[MedicalRecordInput],
        date_window_days: int,
    ) -> Optional[str]:
        """
        将单张票据匹配到某条病历记录。

        匹配规则：
          - bill.hospital 与 medical.hospital_name 模糊匹配（包含关系即可）
          - bill.issue_date 在 medical.startDate ~ medical.endDate 的 ±date_window_days 天范围内
          - hospital 为空或日期无法解析时，跳过该条件
          - 两个条件都满足才算匹配成功

        返回匹配到的 medical_id，未匹配返回 None。
        """
        bill_hospital = getattr(bill, "hospital", None) or ""
        bill_issue_date = getattr(bill, "issue_date", None) or ""

        for medical in medical_records:
            # ── 医院名称匹配 ──────────────────────────────────────
            hospital_match = True
            if bill_hospital and medical.hospital_name:
                # 包含关系即可
                hospital_match = (
                    bill_hospital in medical.hospital_name
                    or medical.hospital_name in bill_hospital
                )
            # 如果 hospital 为空，跳过该条件（默认通过）

            # ── 日期范围匹配 ──────────────────────────────────────
            date_match = True
            if bill_issue_date and medical.startDate and medical.endDate:
                try:
                    bill_dt = _parse_date(bill_issue_date)
                    start_dt = _parse_date(medical.startDate)
                    end_dt = _parse_date(medical.endDate)
                    if bill_dt and start_dt and end_dt:
                        window = timedelta(days=date_window_days)
                        date_match = (start_dt - window) <= bill_dt <= (end_dt + window)
                except (ValueError, OverflowError):
                    # 日期无法解析，跳过该条件
                    date_match = True
            # 如果 issue_date 或 startDate/endDate 为空，跳过日期条件

            if hospital_match and date_match:
                return medical.medical_id

        return None

    @staticmethod
    def _compute_hash(text: str) -> str:
        """返回 SHA1 hex digest 前 16 字符"""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _extract_seq(image_id: str) -> int:
        """
        从 image_id 中提取序号数字，用于排序。
        例如 "img_003" -> 3，"2" -> 2。
        找不到数字时返回 0。
        """
        nums = re.findall(r"\d+", image_id)
        return int(nums[-1]) if nums else 0


# ── 辅助函数 ───────────────────────────────────────────────────────


def _parse_date(date_str: str) -> Optional[datetime]:
    """
    尝试解析日期字符串，支持常见格式。
    解析失败返回 None。
    """
    if not date_str:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y年%m月%d日",
        "%Y.%m.%d",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None
