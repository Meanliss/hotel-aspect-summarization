# Báo Cáo Pipeline LLM ABSA End-to-End - hotel_review1_vi_100plus_llm_v3_full

- Thời điểm tạo report: `2026-06-11 00:15:53`
- Mục tiêu: mô tả đầy đủ pipeline từ dữ liệu review gốc, tách câu, xử lý câu, gán ABSA, repair taxonomy/anchor, rollup evidence, tạo summary, xuất output và metric.
- Nguyên tắc đọc report: số liệu cuối cùng ưu tiên các artifact sau repair ngày 2026-06-10 và trace audit-ready ngày 2026-06-11.

## 1. Executive Summary

Pipeline hiện tại có `1,595,680` ABSA segment cuối cùng từ `1` `data_source`. Tất cả segment đều có cluster assignment sau repair: `1,595,680` / `1,595,680`. File audit cuối cùng nên dùng là `sentence_absa_processing_trace.csv`, không dùng raw `segmentation_trace.csv` để kết luận cluster vì trace cũ còn nhiều cluster field rỗng.

Sentiment tổng thể:

| sentiment   | count   |
|:------------|:--------|
| positive    | 982,556 |
| negative    | 339,268 |
| neutral     | 273,856 |

Aspect tổng thể:

| aspect     | count   |
|:-----------|:--------|
| facility   | 731,646 |
| service    | 350,760 |
| experience | 235,926 |
| amenity    | 257,997 |
| loyalty    | 18,187  |
| branding   | 1,164   |

## 2. Artifact Inventory

Bảng này là danh sách file cần xem khi audit pipeline. Cột `status_or_caution` chỉ rõ file nào là primary, file nào chỉ là trace phụ hoặc aggregate.

| role            | artifact                        | file_pattern                                                              | rows      | purpose                                                                                                                         | status_or_caution                                                                |
|:----------------|:--------------------------------|:--------------------------------------------------------------------------|:----------|:--------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
| primary         | processed_sentences             | hotel_review1_vi_100plus_llm_v3_full_processed_sentences.csv              | 1,595,680 | canonical row-level ABSA units; authoritative after repair                                                                      | OK; use as source for rebuild                                                    |
| secondary_trace | segmentation_trace              | hotel_review1_vi_100plus_llm_v3_full_segmentation_trace.csv               | 1,595,680 | pre-segmentation/trace companion                                                                                                | Still not final cluster trace unless synced from processed_sentences             |
| primary         | cluster_evidence                | hotel_review1_vi_100plus_llm_v3_full_cluster_evidence.csv                 | 131,104   | hotel/aspect/sentiment/cluster evidence rollup                                                                                  | OK counts; FAC15 used for Location & Surroundings                                |
| primary         | cluster_sentiment_summary       | hotel_review1_vi_100plus_llm_v3_full_cluster_sentiment_summary.csv        | 131,104   | one row per hotel/aspect/sentiment/cluster with its own summary                                                                 | NEW correct per-cluster summary layer                                            |
| primary         | cluster_three_sentiment_summary | hotel_review1_vi_100plus_llm_v3_full_cluster_three_sentiment_summary.csv  | 61,799    | one row per hotel/aspect/cluster with positive/negative/neutral summaries                                                       | NEW correct three-sentiment cluster summary view                                 |
| aggregate       | cluster_sentence_summary        | hotel_review1_vi_100plus_llm_v3_full_cluster_sentence_summary.csv         | 23,123    | one aggregate row per hotel/aspect/sentiment summarizing top clusters                                                           | Do not treat this as per-cluster summary                                         |
| primary         | aspect_summary                  | hotel_review1_vi_100plus_llm_v3_full_aspect_summary_from_cluster.csv      | 8,834     | summary per hotel/aspect with 3 sentiments                                                                                      | OK counts after repair                                                           |
| primary         | final_summary                   | hotel_review1_vi_100plus_llm_v3_full_final_summary.csv                    | 10,494    | aspect rows plus all_aspects rows with cluster JSON                                                                             | OK counts after repair                                                           |
| primary         | hotel_overall_summary           | hotel_review1_vi_100plus_llm_v3_full_hotel_overall_summary.csv            | 1,660     | overall hotel summary                                                                                                           | OK                                                                               |
| primary         | output                          | hotel_review1_vi_100plus_llm_v3_full_output.csv                           | 1,660     | wide hotel-level output                                                                                                         | OK                                                                               |
| primary         | summary_metrics                 | hotel_review1_vi_100plus_llm_v3_full_summary_metrics.csv                  | 8,834     | ROUGE/coverage metrics for aspect summary                                                                                       | BERTScore disabled                                                               |
| primary         | final_summary_metrics           | hotel_review1_vi_100plus_llm_v3_full_final_summary_metrics.csv            | 8,834     | ROUGE/coverage metrics for final summary                                                                                        | BERTScore disabled                                                               |
| primary         | cluster_taxonomy                | hotel_review1_vi_100plus_llm_v3_full_cluster_taxonomy.csv                 | 51        | taxonomy from Excel repaired for unique code                                                                                    | OK: FAC2 Room Comfort, FAC15 Location & Surroundings                             |
| audit           | cluster_sentiment_summary_stats | hotel_review1_vi_100plus_llm_v3_full_cluster_sentiment_summary_stats.json |           | stats for per-cluster summary rebuild                                                                                           | Generated 2026-06-10                                                             |
| audit           | sentence_absa_processing_trace  | hotel_review1_vi_100plus_llm_v3_full_sentence_absa_processing_trace.csv   | 1,595,680 | end-to-end row trace: original review, split sentence, processed sentence, final ABSA sentiment and repaired cluster assignment | Generated 2026-06-11; use this instead of raw segmentation_trace for final audit |

Row count theo manifest phân tích:

| artifact                        | rows      |
|:--------------------------------|:----------|
| processed_sentences             | 1,595,680 |
| segmentation_trace              | 1,595,680 |
| cluster_evidence                | 131,104   |
| cluster_sentence_summary        | 23,123    |
| cluster_sentiment_summary       | 131,104   |
| cluster_three_sentiment_summary | 61,799    |
| aspect_summary                  | 8,834     |
| aspect_summary_from_cluster     | 8,834     |
| hotel_overall_summary           | 1,660     |
| final_summary                   | 10,494    |
| output                          | 1,660     |
| summary_metrics                 | 8,834     |
| final_summary_metrics           | 8,834     |
| cluster_taxonomy                | 51        |
| sentence_absa_processing_trace  | 1,595,680 |

## 3. Pipeline Flow Từ Đầu Đến Cuối

### 3.1 Input Review Và Metadata

Các dòng xử lý giữ lại metadata nguồn gồm `source_file`, `entity_id`, `data_source`, `hotel_id`, `review_index`, `sentence_id` và `aspect_segment_id`. `hotel_id` là khóa chính để rollup theo khách sạn; `aspect_segment_id` là khóa chi tiết để truy vết một unit ABSA cụ thể.

Phân bổ khách sạn theo nguồn trong output:

| data_source   | hotel_count   |
|:--------------|:--------------|
| booking       | 1,660         |

Lineage khóa chính xuyên suốt pipeline:

| stage           | key_or_columns                                                                         | meaning                                                |
|:----------------|:---------------------------------------------------------------------------------------|:-------------------------------------------------------|
| review/source   | `source_file`, `entity_id`, `data_source`, `hotel_id`, `review_index`                  | Giữ nguồn gốc review và khách sạn.                     |
| sentence split  | `sentence_id`, `source_sentence`, `preseg_sentence`                                    | Truy vết câu được tách từ review gốc.                  |
| aspect segment  | `aspect_segment_id`, `aspect_segment_text`, `segment_text`                             | Một câu có thể sinh nhiều segment ABSA theo aspect.    |
| processing      | `shortened_sentence`, `processed_sentence`, `normalized_text_vi`, `normalized_text_en` | Lưu các biến thể câu trước khi classification/summary. |
| ABSA assignment | `aspect`, `sentiment`, `cluster_code`, `cluster_label`, `cluster_assignment_source`    | Kết quả gán aspect, sentiment và cluster cuối.         |
| rollup          | `hotel_id + aspect + sentiment + cluster_code`                                         | Khóa gom evidence và summary theo cluster.             |

### 3.2 Tách Câu Và Tạo Segment

Raw `segmentation_trace.csv` có cùng số dòng với `processed_sentences.csv`, tức `1,595,680` dòng, và lưu các trường quan trọng như `source_review`, `source_sentence`, `preseg_sentence`, `aspect_segment_text`, `segment_text`. Tuy nhiên file này không còn đủ tin cậy cho cluster cuối vì `cluster_code` trong raw trace bị rỗng ở `1,575,187` dòng.

Vì vậy đã tạo thêm `sentence_absa_processing_trace.csv`: file này lấy phần tách câu từ `segmentation_trace.csv`, nhưng lấy ABSA/sentiment/cluster từ `processed_sentences.csv` sau repair. Đây là file truy vết cuối cùng để chứng minh pipeline end-to-end.

### 3.3 Chuẩn Hóa Và Xử Lý Câu

`processed_sentences.csv` là canonical row-level ABSA output. File này lưu đầy đủ chuỗi biến đổi: `source_sentence` -> `shortened_sentence` -> `processed_sentence` -> `normalized_text_vi` / `normalized_text_en` -> `classification_text` -> `summary_text`. Các cột này cho phép kiểm tra câu có bị rút gọn/chỉnh sửa trước khi gán aspect, sentiment và cluster hay không.

### 3.4 Gán ABSA Và Sentiment

Mỗi row cuối cùng có `aspect`, `sentiment`, `sentiment_confidence`, `cluster_code`, `cluster_label`, `cluster_descriptors`, `cluster_assignment_confidence` và `cluster_assignment_source`. Sentiment totals trong trace mới khớp với `processed_sentences`.

| sentiment   | count   |
|:------------|:--------|
| positive    | 982,556 |
| negative    | 339,268 |
| neutral     | 273,856 |

### 3.5 Taxonomy Và Repair FAC2/FAC15

Audit cũ phát hiện lỗi taxonomy thật: `FAC2` bị dùng đồng thời cho `Room Comfort` và `Location & Surroundings`. Trạng thái hiện tại đã sửa theo nguyên tắc: `FAC2 = Room Comfort`, `FAC15 = Location & Surroundings`, `FAC14 = View & Surrounding Scenery`.

Taxonomy hiện tại theo aspect:

| aspect     |   cluster_count |
|:-----------|----------------:|
| facility   |              15 |
| amenity    |              12 |
| service    |               9 |
| experience |               6 |
| branding   |               5 |
| loyalty    |               4 |

Repair audit ghi nhận các nhóm thay đổi chính:

| repair_group                            | row_count   |
|:----------------------------------------|:------------|
| changed_total                           | 292,419     |
| changed_location_code                   | 177,067     |
| changed_location_current_anchor         | 99,337      |
| changed_view_current_anchor             | 9,746       |
| changed_location_context_low_confidence | 5,730       |
| changed_view_context_low_confidence     | 460         |
| changed_comfort_not_location            | 79          |

Nguyên tắc repair quan trọng: context chỉ hỗ trợ khi câu hiện tại mơ hồ hoặc confidence thấp; không override câu đã có anchor rõ như breakfast, staff, room cleanliness, location hoặc view.

### 3.6 Cluster Evidence Rollup

`cluster_evidence.csv` có `131,104` dòng, là rollup theo `hotel_id + aspect + sentiment + cluster`. Tổng `count` cộng lại bằng toàn bộ ABSA segment: `1,595,680`.

Top 20 cluster toàn bộ dataset:

| aspect     | cluster_code   | cluster_label              | count   |
|:-----------|:---------------|:---------------------------|:--------|
| service    | SER1           | Staff Friendliness         | 283,049 |
| facility   | FAC15          | Location & Surroundings    | 282,134 |
| facility   | FAC1           | Room Cleanliness           | 186,497 |
| experience | EXP1           | Overall Satisfaction       | 157,085 |
| amenity    | AM2            | Breakfast Quality          | 76,156  |
| amenity    | AM1            | Wifi Quality               | 62,313  |
| facility   | FAC14          | View & Surrounding Scenery | 45,761  |
| experience | EXP3           | Value for Money            | 42,610  |
| facility   | FAC10          | Spaciousness               | 42,267  |
| amenity    | AM12           | Food & Beverage Quality    | 41,289  |
| facility   | FAC3           | Bathroom Condition         | 38,037  |
| amenity    | AM3            | Pool                       | 33,000  |
| facility   | FAC2           | Room Comfort               | 30,213  |
| facility   | FAC11          | Noise Condition            | 23,281  |
| facility   | FAC4           | Bed Quality                | 20,746  |
| service    | SER8           | Hospitality                | 17,319  |
| service    | SER3           | Check-in/Check-out         | 17,163  |
| facility   | FAC12          | Air Conditioning           | 16,928  |
| facility   | FAC8           | Building Condition         | 16,724  |
| facility   | FAC5           | Interior Design            | 16,557  |

### 3.7 Cluster Summary Đúng Theo 3 Sentiment

`cluster_sentence_summary.csv` chỉ là aggregate theo `hotel_id + aspect + sentiment`, không phải per-cluster summary. File đúng cho yêu cầu summary theo từng cluster là `cluster_sentiment_summary.csv` và `cluster_three_sentiment_summary.csv`.

| file                                | rows    | meaning                                                                                                |
|:------------------------------------|:--------|:-------------------------------------------------------------------------------------------------------|
| cluster_sentiment_summary.csv       | 131,104 | Một dòng cho mỗi hotel/aspect/sentiment/cluster, có `cluster_summary` riêng.                           |
| cluster_three_sentiment_summary.csv | 61,799  | Một dòng cho mỗi hotel/aspect/cluster, tách `positive_summary`, `negative_summary`, `neutral_summary`. |

Lưu ý: summary cụm hiện tại là deterministic local summary sinh từ descriptors và evidence samples trong `cluster_evidence.csv`; chưa phải LLM rewrite. Điều này giúp trace đúng với evidence, nhưng văn phong vẫn bị ràng buộc bởi dữ liệu descriptor/evidence đầu vào.

### 3.8 Aspect Summary, Final Summary Và Output

`aspect_summary_from_cluster.csv` có `8,834` dòng và là summary theo hotel/aspect, lấy dữ liệu từ cluster evidence sau repair. `final_summary.csv` có `10,494` dòng, gồm các dòng aspect và dòng `all_aspects` cho từng khách sạn. `hotel_overall_summary.csv` và `output.csv` đều có `1,660` dòng, tương ứng 1 dòng cho mỗi khách sạn.

### 3.9 Metrics

`summary_metrics.csv` và `final_summary_metrics.csv` đang tính ROUGE/coverage. BERTScore không khả dụng trong lần chạy này nên các trường BERTScore bằng 0 và `bertscore_available=False`.

| metric              |   aspect_summary_mean |   final_summary_mean |
|:--------------------|----------------------:|---------------------:|
| bertscore_f1        |                0      |               0      |
| bertscore_precision |                0      |               0      |
| bertscore_recall    |                0      |               0      |
| coverage_score      |                0.3019 |               0.3006 |
| rouge1_recall       |                0.2112 |               0.21   |
| rouge2_recall       |                0.1319 |               0.1318 |
| rouge_l_recall      |                0.1259 |               0.1257 |

### 3.10 Validation Gates

| gate                                     | check                                                                     | result                                                    |
|:-----------------------------------------|:--------------------------------------------------------------------------|:----------------------------------------------------------|
| Row-level trace đầy đủ                   | sentence_absa_processing_trace row_count == processed_sentences row_count | PASS: 1,595,680 == 1,595,680                              |
| Không thiếu cluster cuối                 | cluster_assigned_rows == trace row_count                                  | PASS: 1,595,680 / 1,595,680                               |
| Sentiment totals khớp                    | trace sentiment counts == processed/final rollup totals                   | PASS: positive 982,556; negative 339,268; neutral 273,856 |
| Cluster evidence khớp tổng ABSA          | sum(cluster_evidence.count) == processed row_count                        | PASS: 1,595,680                                           |
| Taxonomy FAC2/FAC15                      | FAC2 chỉ là Room Comfort; FAC15 là Location & Surroundings                | PASS trong taxonomy hiện tại                              |
| Per-cluster summary đúng tầng            | cluster_sentiment_summary rows == cluster_evidence rows                   | PASS: 131,104 == 131,104                                  |
| Trace cũ không dùng làm kết luận cluster | raw segmentation_trace có blank cluster_code lớn                          | CAUTION: 1,575,187 dòng blank cluster_code                |

## 4. Deep Dive Hotel 10638

Hotel `10638` đã có bộ file corrected riêng để đọc theo cluster. Top cluster dưới đây lấy từ `hotel_10638_cluster_sentiment_summary_corrected.csv`, tức summary đúng theo từng cluster-sentiment.

|   hotel_id | aspect     | sentiment   | cluster_code   | cluster_label              |   count |   avg_confidence | cluster_summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|-----------:|:-----------|:------------|:---------------|:---------------------------|--------:|-----------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      10638 | facility   | positive    | FAC15          | Location & Surroundings    |     512 |           0.9465 | Ở nhóm cơ sở vật chất liên quan đến vị trí và khu vực xung quanh, phản hồi tích cực cho thấy khách đánh giá tốt điểm này, nổi bật ở các cảm nhận như thuận tiện, tốt, gần trung tâm, gần biển, vui vẻ, gần bãi tắm và giá tốt; các câu tiêu biểu nhắc đến Vị trí thuận tiện và Vị trí tốt. Nhìn chung, 512 câu trong cluster này phản ánh đây là một điểm mạnh lặp lại của khách sạn.                                                                                                                        |
|      10638 | service    | positive    | SER1           | Staff Friendliness         |     435 |           0.9436 | Ở nhóm dịch vụ liên quan đến sự thân thiện của nhân viên, phản hồi tích cực cho thấy khách đánh giá tốt điểm này, nổi bật ở các cảm nhận như thân thiện, phục vụ tốt, dễ thương, hỗ trợ nhiệt tình, lễ phép, chu đáo và nhiệt tình; các câu tiêu biểu nhắc đến Nhân viên cực kỳ thân thiện và Phòng nghỉ, nhà ăn phục vụ tốt. Nhìn chung, 435 câu trong cluster này phản ánh đây là một điểm mạnh lặp lại của khách sạn.                                                                                     |
|      10638 | facility   | positive    | FAC1           | Room Cleanliness           |     283 |           0.9445 | Ở nhóm cơ sở vật chất liên quan đến độ sạch phòng, phản hồi tích cực cho thấy khách đánh giá tốt điểm này, nổi bật ở các cảm nhận như tốt, sạch sẽ, sáng sủa, đẹp, sạch, tiện nghi và thoáng mát; các câu tiêu biểu nhắc đến Phòng nghỉ tốt và Sạch sẽ. Nhìn chung, 283 câu trong cluster này phản ánh đây là một điểm mạnh lặp lại của khách sạn.                                                                                                                                                           |
|      10638 | facility   | neutral     | FAC15          | Location & Surroundings    |     237 |           0.9372 | Ở nhóm cơ sở vật chất liên quan đến vị trí và khu vực xung quanh, phản hồi trung lập chủ yếu mô tả trạng thái hoặc thông tin khách ghi nhận, với các tín hiệu như dễ thương, trung tâm, không đa dạng, ngon, gần trung tâm, gần bãi tắm và gần quán ăn; các câu đại diện nhắc đến Chủ cực kì dễ thương và Vị trí trung tâm. Nhóm 237 câu này không nghiêng mạnh về khen hay chê, nhưng giúp làm rõ bối cảnh sử dụng của cluster.                                                                             |
|      10638 | facility   | negative    | FAC1           | Room Cleanliness           |     192 |           0.9405 | Ở nhóm cơ sở vật chất liên quan đến độ sạch phòng, phản hồi tiêu cực tập trung vào những bất tiện khách gặp phải, đặc biệt là kém, không đẹp, ẩm ướt, nước chảy ra ngoài, cũ, dơ và ướt át; một số nhận xét cụ thể nhắc rằng Phòng cách âm kém và Nên lưu ý các chi tiết không đẹp ở khu vực cửa sổ trước khi giao phòng. Vì vậy, 192 câu trong cluster này nên được đọc như một nhóm vấn đề cụ thể cần ưu tiên kiểm tra hoặc cải thiện.                                                                     |
|      10638 | facility   | positive    | FAC14          | View & Surrounding Scenery |     188 |           0.9456 | Ở nhóm cơ sở vật chất liên quan đến tầm nhìn và cảnh quan, phản hồi tích cực cho thấy khách đánh giá tốt điểm này, nổi bật ở các cảm nhận như đẹp, tuyệt vời, ngắm bình minh, hơn trông đợi, biển đẹp, có view biển và tốt; các câu tiêu biểu nhắc đến Khách sạn có view đẹp và View đẹp. Nhìn chung, 188 câu trong cluster này phản ánh đây là một điểm mạnh lặp lại của khách sạn.                                                                                                                         |
|      10638 | service    | negative    | SER1           | Staff Friendliness         |     167 |           0.9369 | Ở nhóm dịch vụ liên quan đến sự thân thiện của nhân viên, phản hồi tiêu cực tập trung vào những bất tiện khách gặp phải, đặc biệt là thiếu duyên, gây e ngại, gây ồn, không hài lòng, không linh hoạt, chậm và cần nâng cao; một số nhận xét cụ thể nhắc rằng Nhân viên lễ tân chưa đặt lắm, hơi thiếu cái duyên làm nghề và E ngại không biết làm cách nào để nhân viên phân biệt công dụng. Vì vậy, 167 câu trong cluster này nên được đọc như một nhóm vấn đề cụ thể cần ưu tiên kiểm tra hoặc cải thiện. |
|      10638 | experience | positive    | EXP1           | Overall Satisfaction       |     163 |           0.9097 | Ở nhóm trải nghiệm liên quan đến mức hài lòng tổng thể, phản hồi tích cực cho thấy khách đánh giá tốt điểm này, nổi bật ở các cảm nhận như thích, không quá ồn, yên tĩnh, ngắm bình minh thú vị, tốt, sẽ nghỉ thêm và hài lòng; các câu tiêu biểu nhắc đến Rất thích khách sạn và Nhưng may mắn là khách lưu trú ở khách sạn không quá ồn. Nhìn chung, 163 câu trong cluster này phản ánh đây là một điểm mạnh lặp lại của khách sạn.                                                                        |
|      10638 | amenity    | negative    | AM2            | Breakfast Quality          |     141 |           0.94   | Ở nhóm tiện ích liên quan đến chất lượng bữa sáng, phản hồi tiêu cực tập trung vào những bất tiện khách gặp phải, đặc biệt là thiếu đa dạng, ít lựa chọn, chưa được thưởng thức, cần phong phú hơn, ít món, chưa đa dạng và không ngon; một số nhận xét cụ thể nhắc rằng Đồ ăn sáng đa dạng hơn thì tuyệt vời và Buffe sáng hơi ít lựa chọn. Vì vậy, 141 câu trong cluster này nên được đọc như một nhóm vấn đề cụ thể cần ưu tiên kiểm tra hoặc cải thiện.                                                  |
|      10638 | experience | negative    | EXP1           | Overall Satisfaction       |     128 |           0.9089 | Ở nhóm trải nghiệm liên quan đến mức hài lòng tổng thể, phản hồi tiêu cực tập trung vào những bất tiện khách gặp phải, đặc biệt là kêu, ồn, khó khăn, không tốt, phải đi vòng vo, rất bất tiện và bị bắt buộc phải chạy vòng vo; một số nhận xét cụ thể nhắc rằng Điều hòa đêm kêu hơi ồn và Máy lạnh ồn. Vì vậy, 128 câu trong cluster này nên được đọc như một nhóm vấn đề cụ thể cần ưu tiên kiểm tra hoặc cải thiện.                                                                                     |
|      10638 | facility   | negative    | FAC3           | Bathroom Condition         |     128 |           0.9439 | Ở nhóm cơ sở vật chất liên quan đến tình trạng phòng tắm, phản hồi tiêu cực tập trung vào những bất tiện khách gặp phải, đặc biệt là cũ, dùng làm khăn lau, cần thay mới, thiết kế lạ, ướt, dễ bị ướt và tồn đọng nước; một số nhận xét cụ thể nhắc rằng Khăn nhà tắm đã cũ và Khăn tắm cũ dùng làm khăn lau. Vì vậy, 128 câu trong cluster này nên được đọc như một nhóm vấn đề cụ thể cần ưu tiên kiểm tra hoặc cải thiện.                                                                                 |
|      10638 | experience | positive    | EXP3           | Value for Money            |     116 |           0.9342 | Ở nhóm trải nghiệm liên quan đến giá trị so với chi phí, phản hồi tích cực cho thấy khách đánh giá tốt điểm này, nổi bật ở các cảm nhận như hợp lý, đáng giá tiền, rẻ, sạch sẽ, rộng rãi, ổn và tốt; các câu tiêu biểu nhắc đến Giá hợp lý và Giá rất hợp lý. Nhìn chung, 116 câu trong cluster này phản ánh đây là một điểm mạnh lặp lại của khách sạn.                                                                                                                                                     |
|      10638 | facility   | negative    | FAC15          | Location & Surroundings    |     108 |           0.9435 | Ở nhóm cơ sở vật chất liên quan đến vị trí và khu vực xung quanh, phản hồi tiêu cực tập trung vào những bất tiện khách gặp phải, đặc biệt là khó tìm, chưa ổn, không giống như hình ảnh miêu tả, quá xa, không có, xa hotel và bất tiện; một số nhận xét cụ thể nhắc rằng Vị trí trong hẻm, khó tìm và Chỗ đậu xe chưa ổn. Vì vậy, 108 câu trong cluster này nên được đọc như một nhóm vấn đề cụ thể cần ưu tiên kiểm tra hoặc cải thiện.                                                                    |
|      10638 | amenity    | positive    | AM2            | Breakfast Quality          |      92 |           0.9393 | Ở nhóm tiện ích liên quan đến chất lượng bữa sáng, phản hồi tích cực cho thấy khách đánh giá tốt điểm này, nổi bật ở các cảm nhận như đa dạng, ngon, hợp khẩu vị, vừa miệng, tự túc phù hợp, khá đầy đủ và tuyệt vời; các câu tiêu biểu nhắc đến Ăn sáng đa dạng và Ăn sáng ngon. Nhìn chung, 92 câu trong cluster này phản ánh đây là một điểm mạnh lặp lại của khách sạn.                                                                                                                                  |
|      10638 | experience | neutral     | EXP1           | Overall Satisfaction       |      79 |           0.8911 | Ở nhóm trải nghiệm liên quan đến mức hài lòng tổng thể, phản hồi trung lập chủ yếu mô tả trạng thái hoặc thông tin khách ghi nhận, với các tín hiệu như như quảng cáo, ngủ được, phù hợp người Việt, phù hợp người nước ngoài, ngắn, sớm và có 1 giường đôi; các câu đại diện nhắc đến Như hình quảng cáo và vẫn ngủ được. Nhóm 79 câu này không nghiêng mạnh về khen hay chê, nhưng giúp làm rõ bối cảnh sử dụng của cluster.                                                                               |

## 5. File Nên Dùng Cho Từng Mục Đích

| need                                           | file                                           |
|:-----------------------------------------------|:-----------------------------------------------|
| Audit câu gốc -> câu tách -> câu xử lý -> ABSA | sentence_absa_processing_trace.csv             |
| ABSA canonical row-level                       | processed_sentences.csv                        |
| Rollup evidence theo cluster                   | cluster_evidence.csv                           |
| Summary từng cluster theo sentiment            | cluster_sentiment_summary.csv                  |
| Summary từng cluster có 3 sentiment columns    | cluster_three_sentiment_summary.csv            |
| Summary theo aspect                            | aspect_summary_from_cluster.csv                |
| Summary cuối theo aspect + all aspects         | final_summary.csv                              |
| Output wide theo khách sạn                     | output.csv                                     |
| Metric summary                                 | summary_metrics.csv, final_summary_metrics.csv |
| Taxonomy sau repair                            | cluster_taxonomy.csv                           |

## 6. Caveat Và Rủi Ro Còn Lại

- Raw `segmentation_trace.csv` không nên dùng để kết luận cluster cuối vì còn `1,575,187` dòng rỗng `cluster_code`; dùng `sentence_absa_processing_trace.csv` thay thế.
- Audit 2026-06-09 phản ánh trạng thái trước repair, đặc biệt phần duplicate `FAC2`; report này dùng trạng thái sau repair.
- Cluster summary hiện tại không gọi LLM lại, mà sinh deterministic từ `cluster_evidence`; nếu cần văn phong tự nhiên hơn nữa có thể thêm LLM rewrite pass nhưng phải khóa evidence để không hallucinate.
- Một số descriptor có thể nhiễu do phụ thuộc chất lượng sentence processing ban đầu; cần audit sample định kỳ cho nhóm risky như location/view/room comfort/breakfast/wifi.
- Metrics hiện chỉ là ROUGE/coverage; chưa có BERTScore nên không nên diễn giải như đánh giá semantic đầy đủ.

## 7. Kết Luận

Pipeline hiện đã có đầy đủ artifact để truy vết từ review gốc đến output cuối. Điểm thiếu trước đó là file lưu trữ đầy đủ quá trình tách câu, xử lý câu và gán ABSA sau repair; file này đã được bổ sung bằng `sentence_absa_processing_trace.csv`. Bộ file sau repair hiện nhất quán về tổng dòng, sentiment totals, taxonomy FAC2/FAC15 và rollup cluster evidence.
