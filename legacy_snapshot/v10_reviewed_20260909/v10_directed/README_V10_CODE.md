# Исполняемый пакет v10

Код реализует frozen-протокол
`RESEARCH_PROTOCOL_V10_DIRECTED_VOLATILITY_NETWORKS.md` и пишет новые
артефакты только в каталоги с суффиксом `v10`. Результаты v09 не меняются.

## Зависимости

```bash
conda activate WF
python -m pip install -r v10_directed/requirements_v10.txt
```

Все команды ниже запускаются из корня проекта, где расположен `data_v02`.
Для ограничения памяти рекомендуется сохранять `MALLOC_ARENA_MAX=2`.

## 1. IMOEX и causal residual RV

```bash
python -u v10_directed/download_imoex_10m_v01.py --root data_v02

MALLOC_ARENA_MAX=2 python -u \
  v10_directed/build_market_factor_residuals_v01.py --root data_v02

python -u v10_directed/build_market_factor_residuals_v01.py \
  --root data_v02 --assemble
```

Первый запуск residual builder обрабатывает SECID последовательно и сохраняет
checkpoint каждого актива. После прерывания команда безопасно перезаписывает
только указанные SECID; для ручного шардинга:

```bash
python -u v10_directed/build_market_factor_residuals_v01.py \
  --root data_v02 --secids SBER,GAZP,LKOH
```

Перед residual-графами проверьте:

```bash
cat data_v02/market_factor_v10/market_factor_coverage.csv
```

Обе строки должны иметь `passed=True`. Если gate не пройден, residual/H3
останавливается; заменять IMOEX другим фактором в v10 нельзя.

## 2. Выбор глобальных graph-learner параметров

Сначала проверяется Amendment 01: прогнозные target остаются dynamic top-40,
но их causal train/importance history берется из all-82 RV panel. Этот быстрый
gate выполняется до обучения grid:

```bash
python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage audit-panel
```

По Amendment 02 ожидается ровно 40 active targets в каждом блоке, не менее 32
feasible targets на покрытой дате, `valid_date_rate >= 0.90` и
`median_targets >= 35`. Пороги отдельных моделей 400/80 не снижены. При
провале selection запускать нельзя.

CatBoost-конфигурации можно считать по одной, поэтому убийство процесса не
теряет уже готовые результаты:

```bash
for cfg in cb{01..16}; do
  MALLOC_ARENA_MAX=2 python -u \
    v10_directed/build_directed_predictive_graphs_v01.py \
    --root data_v02 --stage select --learner catboost --config-id "$cfg"
done

for cfg in en{01..15}; do
  MALLOC_ARENA_MAX=2 python -u \
    v10_directed/build_directed_predictive_graphs_v01.py \
    --root data_v02 --stage select --learner linear --config-id "$cfg"
done

python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage finalize-selection
```

При нестандартном расположении frozen split добавьте
`--split-dates ПУТЬ/split_dates.csv` к командам `--stage select`.

ElasticNet сохраняет frozen `max_iter=10000`. Если конкретный fit не сошелся,
он получает `fit_status=nonconverged`, исключается из QLIKE selection и не
может стать победителем. `--force` пересчитывает только переданный config-id,
например после обновления реализации:

```bash
python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage select --learner linear \
  --config-id en01 --targets SBER --force
```

## 3. Построение raw directed graphs

```bash
MALLOC_ARENA_MAX=2 python -u \
  v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage build --learner linear

MALLOC_ARENA_MAX=2 python -u \
  v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage build --learner catboost

python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage pearson

python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage assemble
```

Каждый `block_id × target` является отдельным checkpoint. Для распределения
работы можно передать `--block-id 47` или `--targets SBER,GAZP`. Повторный
запуск пропускает готовые части; `--force` применяется только после исправления
реализационной ошибки.

## 4. Признаки, placebos и stress

```bash
python -u v10_directed/build_directed_graph_features_v01.py \
  --root data_v02 --variant raw
```

Скрипт строит M0–M7 features, lag-20, source-identity, frozen-graph placebo,
continuous `stress_z` и wide source groups для M8.

## 5. Development walk-forward

Сначала создается отдельный coverage-only split по Amendment 03. Скрипт не
читает target, predictions или losses: ранние directed-valid даты используются
как warm-up, первые 60 последующих valid-дней — для выбора ridge alpha, а
оставшиеся даты до `2026-08-10` — как exploratory development test.

```bash
python -u v10_directed/build_directed_split_v01.py --root data_v02

cat data_v02/model_audit/directed_split_audit_v10.csv
```

Ожидаются не менее 15 warm-up дат, 30 warm-up SECID, ровно 60 validation и не
менее 120 development directed-valid дат. Старый v09 split напрямую для v10 HAR selection не
используется: в нем нет общей directed train-выборки до первого validation
origin.

Alpha можно выбирать моделями-шардами. Для всех M0–M7:

```bash
MALLOC_ARENA_MAX=2 python -u \
  v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --stage select
```

Затем строятся validation и development predictions. Каждый model/block
сохраняется сразу:

```bash
MALLOC_ARENA_MAX=2 python -u \
  v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --stage run --period val
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --stage assemble --period val

MALLOC_ARENA_MAX=2 python -u \
  v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --stage run --period development
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --stage assemble --period development

python -u v10_directed/analyze_directed_results_v01.py \
  --root data_v02 --mode development
```

### Stress-MAD implementation correction

If the initial development H2 audit contains only 63 stress dates, install the
implementation-note-02 patch and rebuild only the deterministic raw feature
artifact before rerunning analysis:

```bash
cp data_v02/model_data/directed_graph_features.parquet \
  data_v02/model_data/directed_graph_features.parquet.pre_stress_mad_fix

python -u v10_directed/build_directed_graph_features_v01.py \
  --root data_v02 --variant raw

python -u v10_directed/audit_stress_mad_fix_v01.py \
  --root data_v02

python -u v10_directed/analyze_directed_results_v01.py \
  --root data_v02 --mode development

cat data_v02/benchmark_v10_directed_development/stress_coverage.csv
cat data_v02/benchmark_v10_directed_development/paired_inference_regime.csv
```

M0--M8 and placebo predictions are not refit: none of their feature sets uses
`stress_z`.  See `V10_IMPLEMENTATION_NOTE_02_STRESS_MAD_FIX.md` for disclosure
and scope.

Если нужно запускать модели отдельно, используйте точные id, например:

```bash
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --stage run --period development \
  --models M6_catboost_directed_current,M7_catboost_directed_dynamic
```

`M8_direct_catboost` обучается target-specific на causal all-82 истории внутри
текущего temporal segment и оценивается на directed common sample. Минимум 400
target-history строк сохраняется. Эта история намеренно не фильтруется по
`directed_sample_valid`, поскольку M8 не использует графовые признаки:

```bash
MALLOC_ARENA_MAX=2 python -u \
  v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --stage run --period development \
  --models M8_direct_catboost --block-id 1
```

После smoke-test первого блока повторите команду без `--block-id`; готовый
checkpoint блока 1 будет пропущен.

Coverage по блокам сохраняется в
`benchmark_v10_directed_development/m8_history_coverage_development.csv`.

## 6. Residual robustness H3

Запускается только после успешного coverage gate:

```bash
for cfg in cb{01..16}; do
  MALLOC_ARENA_MAX=2 python -u \
    v10_directed/build_directed_predictive_graphs_v01.py \
    --root data_v02 --variant residual --stage select \
    --learner catboost --config-id "$cfg"
done
for cfg in en{01..15}; do
  MALLOC_ARENA_MAX=2 python -u \
    v10_directed/build_directed_predictive_graphs_v01.py \
    --root data_v02 --variant residual --stage select \
    --learner linear --config-id "$cfg"
done

python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --variant residual --stage finalize-selection
python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --variant residual --stage build --learner linear
python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --variant residual --stage build --learner catboost
python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --variant residual --stage assemble
python -u v10_directed/build_directed_graph_features_v01.py \
  --root data_v02 --variant residual
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --variant residual --stage select
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --variant residual --stage run --period development
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --variant residual --stage assemble --period development
```

Повторный запуск analysis автоматически добавит `residual_robustness.csv`.

## 7. Заморозка и новый holdout

После завершения development и до просмотра будущих исходов:

```bash
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --stage freeze
```

После появления новых данных признаки и графовые состояния обновляются теми же
скриптами, затем прогнозы сохраняются без target/loss:

```bash
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --stage run --period confirmatory
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --stage assemble --period confirmatory
```

`analyze_directed_results_v01.py --mode confirmatory` откажется раскрывать
метрики до 120 valid forecast dates, кроме заранее зафиксированного календарного
ограничения 2027-03-31.

## Быстрая проверка установки

```bash
python -m py_compile v10_directed/*.py
python v10_directed/build_directed_predictive_graphs_v01.py --help
python v10_directed/train_directed_har_benchmark_v01.py --help
```
