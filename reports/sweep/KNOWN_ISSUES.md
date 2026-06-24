# Sweep — known issues & current state (2026-06-23)

Trạng thái dừng giữa chừng của cuộc sweep siêu tham số (threshold + token).
Đọc kèm memory: `sweep-fixed-denominator-methodology`, `hasos-dir-mapping-confirmed`.

## Tiến độ hiện tại

| Phần | Trạng thái |
|---|---|
| SPACE threshold (M2/M3/M4 × 5 giá trị) | ✅ **15/15 cells xong, ĐÚNG** — đã lên web `/optimality` |
| HASOS threshold (M2/M3/M4 × ≤0.005) | 🔄 **đang chạy tiếp sau khi fix M2 hierarchical parent-pass** |
| Token phase (cả 2 dataset) | ⏳ chưa chạy |

SPACE đã chứng minh xong: threshold "giữ toàn pool" (0.0082) là tối ưu;
siết lại làm ROUGE-1 **và** coverage sụp đồng thời (đơn điệu). Verdict
"Default is optimal" cho cả M2/M3/M4.

## LỖI #1 — M2 HASOS child-only layout từng score ra 0 (FIX ĐÃ VERIFY)

**Triệu chứng cũ:** mọi cell `rouge_m2_hasos_threshold_*.json` child-only từng cho
R1=R2=RL=0, coverage=0 ở tất cả 4 parent aspect — dù output dir có đủ 823 file.

**Nguyên nhân gốc (đã xác định chắc chắn):**
- Harness sweep cố tình bỏ `--hierarchical` để tiết kiệm thời gian → chỉ sinh
  layout sub-aspect code `FAC_BATH/`, `AM_ENT/`, ...
- Scorer M2 (`--parent_dir` → `system_text_parent_dir`) đi tìm
  `<dir>/FACILITY/<file>`, `<dir>/AMENITY/<file>` ... → **không tồn tại** → text rỗng → ROUGE 0.
- SPACE M2 KHÔNG dính vì subdir SPACE trùng gold key (`building/` == "building").
  HASOS subdir là code (`FAC_BATH`) ≠ parent key (`facility`).
- M3/M4 HASOS KHÔNG dính: resolver `senti_dir` đã aggregate code→parent qua `code2group`.
  → chỉ **M2 HASOS** lỗi.

**Bằng chứng (diag, đã xoá file tạm):**
- Baseline M2 HASOS thật = dir `outputs/space_hasos_threshold_full_abstractive_quality_parent`
  (có subdir FACILITY/AMENITY...), score `--parent_dir` → **R1 0.20015** (khớp số đã chốt 0.2002).
- Dir sweep (chỉ child codes) score `--parent_dir` → rỗng (0).
- Dir sweep score như `--run_dir` (aggregate code→parent) → R1 0.209 (gần đúng nhưng
  KHÁC baseline vì baseline dùng parent-pass abstractive, không phải nối child).

**Fix đã áp và đã verify trong `scripts/sweep_params.py`:**
- CONFIG thêm cờ `m2_hierarchical`: SPACE=False, HASOS=True.
- `synthesize()`: nếu `m2 and m2_hierarchical` → thêm `--hierarchical` (chạy parent pass,
  sinh dir `{out_run_id}_parent` chứa file FACILITY/AMENITY/SERVICE/EXPERIENCE).
- `out_dirs()`: M2 HASOS trỏ `--parent_dir` vào `{out_run_id}_parent`.
- Smoke-test `T=0.005`: fixed-denominator R1=**0.19661**, khớp baseline parent-dir fixed R1=**0.19689** (sai khác 0.00028 do generate lại).

**Việc còn phải làm để đóng Phase A HASOS:**
1. Đợi batch đang chạy xong: `sweep_params.py --dataset hasos --phase threshold --grid 0.0,0.0025,0.004 --methods m2,m3,m4`.
2. Sau đó build lại `threshold_hasos_summary.json`, `sweep.json`, và kiểm tra đường cong HASOS.
3. Nếu coverage/ROUGE cho T<0.005 thấp hơn T=0.005 → kết luận default HASOS threshold 0.005 là tối ưu trong hướng siết.

## LỖI #2 — chi phí thời gian HASOS (không phải bug, chỉ là ràng buộc)

HASOS có 507–823 child file/cell (SPACE chỉ 330), + giờ thêm parent pass cho M2.
12 cell HASOS threshold ≈ 2+ giờ GPU. Pool HASOS cap ở 0.005 (điểm tối đa = 0.00500),
nên chỉ test được hướng SIẾT (≤0.005); muốn test RỘNG hơn 0.005 phải rerun SemAE
(`aspect_inference.py`) — chưa làm.

## LỖI #3 — token phase chưa kiểm chứng layout HASOS M2

`phase_token_abstractive()` cũng gọi `out_dirs()` nên đã hưởng fix #1, NHƯNG chưa chạy
thực tế lần nào. Khi chạy token phase cho HASOS M2 nhớ verify lại cell đầu tiên.

## Sạch sẽ

- Mọi tiến trình python synthesis đã bị kill.
- File scratch trong `reports/sweep/` (`_log_*`, `_sanity_*`, `_diag_*`, `_chk_*`, `_run_*`)
  KHÔNG commit (xem .gitignore bổ sung). Chỉ commit cell hợp lệ + summary + scripts + web.
