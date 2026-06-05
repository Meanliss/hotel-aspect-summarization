# SemAE SPACE Full Training Numerical Guide

Mục tiêu của guide này là train lại SPACE/HASOS theo cách không nổ gradient,
không train mù bằng skip batch hàng loạt, và luôn giữ đủ artifact để inference
được về sau.

## 1. Vì sao lần trước dễ hỏng

Checkpoint `.pt` không chứa SentencePiece tokenizer. Nếu chỉ sync checkpoint mà
không sync `data/sentencepiece/space_unigram_32k.model`, checkpoint gần như
không dùng inference an toàn được.

Vấn đề numerical chính nằm ở normalize trong quantizer:

```python
x / x.norm(...)
```

Nếu vector gần zero, norm quá nhỏ có thể làm NaN/Inf lan vào loss/grad. Với full
11k entities, xác suất gặp batch xấu cao hơn subset 500/2k rất nhiều.

Không được dùng `--skip_nonfinite` để train full. Nếu skip lên hàng chục nghìn
batch mỗi epoch thì checkpoint không đáng tin, dù process vẫn chạy.

## 2. Cấu hình ổn định khuyến nghị

Dùng cấu hình này làm default:

```bash
--lr 0.0001
--grad_clip 1.0
--safe_normalize
--diagnose_nonfinite
--nonfinite_policy fail
--kmeans_bad_vector_policy fail
--no_eval
```

Ý nghĩa:

- `--lr 0.0001`: hạ LR từ upstream `1e-3` xuống `1e-4`, giảm bùng gradient.
- `--grad_clip 1.0`: clip gradient nhưng không che NaN/Inf thật.
- `--safe_normalize`: clamp norm bằng epsilon trước khi chia.
- `--diagnose_nonfinite`: log batch/entity/param đầu tiên bị NaN/Inf.
- `--nonfinite_policy fail`: gặp NaN/Inf thì dừng để sửa nguyên nhân, không skip mù.
- `--kmeans_bad_vector_policy fail`: nếu input K-means có zero/NaN/Inf thì dừng.

## 3. Smoke test trước khi train full

Chạy nhỏ để xác nhận env và data sạch:

```bash
cd /workspace/SemAE
export PYTHONPATH=/workspace/SemAE/src:${PYTHONPATH:-}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

python src/train.py \
  --data data/space/json/space_train.json \
  --sentencepiece data/sentencepiece/space_unigram_32k.model \
  --run_id space_smoke_200x1 \
  --gpu 0 \
  --epochs 1 \
  --max_num_entities 200 \
  --batch_size 5 \
  --lr 0.0001 \
  --grad_clip 1.0 \
  --safe_normalize \
  --diagnose_nonfinite \
  --nonfinite_policy fail \
  --kmeans_bad_vector_policy fail \
  --no_eval \
  --savedir /workspace/SemAE/checkpoints/space_smoke_200x1 \
  --save_every 1 2>&1 | tee logs/space_smoke_200x1.log
```

Accept:

- Không `Traceback`.
- Không `NONFINITE_DIAGNOSTIC`.
- `EPOCH_DIAGNOSTIC` có:
  - `skipped_batches=0`
  - `bad_loss_batches=0`
  - `bad_grad_batches=0`

## 4. Run trung gian trước full

Trước khi đốt tiền full 11k, chạy 2k x 4 hoặc 2k x 8:

```bash
python src/train.py \
  --data data/space/json/space_train.json \
  --sentencepiece data/sentencepiece/space_unigram_32k.model \
  --run_id space_2k_stable_8 \
  --gpu 0 \
  --epochs 8 \
  --max_num_entities 2000 \
  --batch_size 5 \
  --d_model 320 \
  --codebook_size 1024 \
  --nlayers 3 \
  --internal_nheads 4 \
  --output_nheads 8 \
  --d_ff 512 \
  --warmup_epochs 4 \
  --lr 0.0001 \
  --lr_decay 0.9 \
  --label_smoothing 0.1 \
  --commitment_cost 1.0 \
  --l1_cost 1000.0 \
  --entropy_cost 0.00005 \
  --grad_clip 1.0 \
  --safe_normalize \
  --diagnose_nonfinite \
  --nonfinite_policy fail \
  --kmeans_bad_vector_policy fail \
  --no_eval \
  --savedir /workspace/SemAE/checkpoints/space_2k_stable_8 \
  --save_every 1 2>&1 | tee logs/space_2k_stable_8.log
```

Accept:

- Epoch 1-4 clean.
- Qua được K-means ở epoch 4.
- Có checkpoint:
  - `space_2k_stable_8_1_model.pt`
  - ...
  - `space_2k_stable_8_4pkm_model.pt`
  - ...
  - `space_2k_stable_8_8_model.pt`
- `bad_loss_batches=0`, `bad_grad_batches=0`, `skipped_batches=0`.

## 5. Full 11k x 20 command

Chỉ chạy full sau khi 2k clean.

```bash
python src/train.py \
  --data data/space/json/space_train.json \
  --sentencepiece data/sentencepiece/space_unigram_32k.model \
  --run_id space_full_11402x20_stable \
  --gpu 0 \
  --epochs 20 \
  --batch_size 5 \
  --d_model 320 \
  --codebook_size 1024 \
  --nlayers 3 \
  --internal_nheads 4 \
  --output_nheads 8 \
  --d_ff 512 \
  --warmup_epochs 4 \
  --lr 0.0001 \
  --lr_decay 0.9 \
  --label_smoothing 0.1 \
  --commitment_cost 1.0 \
  --l1_cost 1000.0 \
  --entropy_cost 0.00005 \
  --grad_clip 1.0 \
  --safe_normalize \
  --diagnose_nonfinite \
  --nonfinite_policy fail \
  --kmeans_bad_vector_policy fail \
  --no_eval \
  --savedir /workspace/SemAE/checkpoints/space_full_11402x20_stable \
  --save_every 1 2>&1 | tee logs/space_full_11402x20_stable.log
```

Không truyền `--max_num_entities` nghĩa là full thật.

## 6. Nếu full fail thì xử lý thế nào

Nếu fail bằng `NONFINITE_DIAGNOSTIC loss`:

- Forward loss đã NaN/Inf.
- Cần xem batch ids trong log.
- Kiểm tra review quá dài/rỗng/vector norm nhỏ.
- Không resume bằng cách skip.

Nếu fail bằng `NONFINITE_DIAGNOSTIC grad`:

- Forward finite, nhưng backward sinh grad NaN/Inf.
- Xem `first_bad` để biết parameter/layer đầu tiên.
- Thử giảm LR tiếp:

```bash
--lr 0.00005
```

Nếu fail ở K-means:

- Log sẽ có `KMEANS_VECTOR_DIAGNOSTIC total=... bad=...`.
- Nếu `bad` nhỏ, có thể chạy diagnostic riêng với:

```bash
--kmeans_bad_vector_policy filter --kmeans_max_bad_vectors 10
```

Nhưng chỉ dùng `filter` khi bad vector rất ít và đã ghi rõ trong report. Nếu
bad nhiều, dừng, không train tiếp.

## 7. Monitor bắt buộc

Trong khi train, theo dõi:

```bash
tail -f logs/space_full_11402x20_stable.log

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv
```

Tìm các chuỗi nguy hiểm:

```bash
grep -E "NONFINITE_DIAGNOSTIC|Traceback|RuntimeError|out of memory|CUDA out of memory|KMEANS_VECTOR_DIAGNOSTIC" logs/space_full_11402x20_stable.log
```

Mỗi epoch phải có:

```text
EPOCH_DIAGNOSTIC epoch=N finite_batches=... skipped_batches=0 bad_loss_batches=0 bad_grad_batches=0 ...
```

## 8. Sync artifact đúng cách

Sau mỗi epoch hoặc ít nhất sau khi dừng, sync cả folder này:

```bash
/workspace/SemAE/checkpoints/<run_id>/
/workspace/SemAE/logs/<run_id>.log
/workspace/SemAE/data/sentencepiece/space_unigram_32k.model
/workspace/SemAE/data/sentencepiece/space_unigram_32k.vocab
```

Ví dụ local:

```powershell
$HostIp = "<ip>"
$Port = <direct_22_port>
$RunId = "space_full_11402x20_stable"

scp -P $Port -r root@${HostIp}:/workspace/SemAE/checkpoints/$RunId D:\KHDL\LuanAn\tesing\models\
scp -P $Port root@${HostIp}:/workspace/SemAE/logs/$RunId.log D:\KHDL\LuanAn\tesing\logs\
scp -P $Port root@${HostIp}:/workspace/SemAE/data/sentencepiece/space_unigram_32k.model D:\KHDL\LuanAn\tesing\data\sentencepiece\
scp -P $Port root@${HostIp}:/workspace/SemAE/data/sentencepiece/space_unigram_32k.vocab D:\KHDL\LuanAn\tesing\data\sentencepiece\
```

Không destroy Vast instance trước khi kiểm local có đủ:

```text
models/<run_id>/<run_id>_<epoch>_model.pt
data/sentencepiece/space_unigram_32k.model
data/sentencepiece/space_unigram_32k.vocab
logs/<run_id>.log
```

## 9. Inference sau khi có model

Chỉ chạy inference khi tokenizer tồn tại:

```bash
test -f data/sentencepiece/space_unigram_32k.model
```

Sau đó:

```bash
python scripts/run_space_hasos_aspect_parallel.py \
  --model checkpoints/space_full_11402x20_stable/space_full_11402x20_stable_20_model.pt \
  --run_id space_hasos_full_e20 \
  --num_shards 4 \
  --gpu 0 \
  --max_tokens 40 \
  --sentiment_split

python scripts/export_space_hasos_lines.py --run_id space_hasos_full_e20
python scripts/summarize_aspect_outputs.py --run_id space_hasos_full_e20
python scripts/score_semae_run.py --run_id space_hasos_full_e20
python scripts/build_space_hasos_report_pptx.py --run_id space_hasos_full_e20
```

## 10. Rule cuối

Checkpoint chỉ được coi là usable khi:

- Có tokenizer khớp.
- `bad_loss_batches=0`.
- `bad_grad_batches=0`.
- `skipped_batches=0`.
- Không OOM.
- K-means qua sạch hoặc bad vectors được giải thích rõ.
- Checkpoint và tokenizer đã sync local trước khi destroy instance.
