# Исследовательский протокол v10

## Направленные предиктивные сети и прогноз realized variance на MOEX

**Статус:** `FROZEN BEFORE IMPLEMENTATION`  
**Дата фиксации:** 2026-08-18  
**Предыдущий этап:** v09 HAR-X Pearson/dCor  
**Основной горизонт:** следующий торговый день  
**Основная функция потерь:** QLIKE  

---

## 0. Назначение протокола

Цель v10 — проверить не очередную модель ради улучшения метрики, а новое
экономическое определение финансовой связи:

> содержит ли прошлое актива-источника дополнительную нелинейную информацию о
> будущей волатильности актива-получателя после учета собственной истории
> получателя и общего рыночного фактора?

В v09 ребра отражали ненаправленную зависимость доходностей, измеренную Pearson
или distance correlation. В v10 основным новым объектом является направленный
предиктивный граф. Ребро `j -> i` существует, если прошлые признаки актива `j`
улучшают out-of-sample прогноз `RV[i, t+1]` относительно информации, уже
доступной модели.

Протокол одновременно решает три задачи:

1. сравнивает ненаправленные dependence-графы с направленными predictive-графами;
2. проверяет, усиливается ли ценность динамики графа в стрессовом рыночном режиме;
3. отделяет общий market mode от локальной передачи волатильности через
   residual/idiosyncratic robustness-анализ.

Нулевой или отрицательный результат является допустимым итогом.

---

## 1. Что уже известно из v09

v09 остается закрытым завершенным этапом и не переоценивается в рамках v10.

- Target: next-day realized variance.
- Основной test: 386 дат, 15 435 наблюдений `date x SECID`.
- Pearson Dynamic немного улучшал Pearson Current:
  `mean daily Delta QLIKE = 0.001660`, HAC `p = 0.1435`.
- dCor Dynamic улучшал dCor Current:
  `mean daily Delta QLIKE = 0.001342`, nominal HAC `p = 0.0480`, но эффект не
  сохранялся после Holm correction.
- Pearson Dynamic и dCor Dynamic статистически не различались.
- GNN temporal-order placebo не дал убедительных доказательств, что нейросеть
  использует порядок прошлых графов.

Следовательно, v10 не формулируется как попытка подтвердить уже доказанный
эффект. Рабочая исходная позиция: в v09 обнаружен небольшой suggestive signal,
которому требуется более содержательное определение ребра и новая
confirmatory-выборка.

Период по 2026-08-10 включительно уже просмотрен. Все результаты v10 на этом
периоде автоматически считаются `development/exploratory`.

---

## 2. Исследовательские вопросы и гипотезы

### 2.1. Основной вопрос

Улучшает ли динамика направленного nonlinear predictive graph прогноз
next-day RV по сравнению с тем же направленным графом, наблюдаемым только в
текущем состоянии?

Основной контраст:

```text
QLIKE(CatBoost-Directed-Current)
    - QLIKE(CatBoost-Directed-Dynamic)
```

Положительная разность означает преимущество Dynamic.

### 2.2. Семейство основных гипотез

#### H1a — ценность изменения направленного графа

```text
CatBoost-Directed-Dynamic < CatBoost-Directed-Current
```

по ожидаемой дневной QLIKE.

#### H1b — ценность направленного nonlinear graph относительно Pearson

```text
CatBoost-Directed-Dynamic < Pearson-Dynamic-Blocked
```

#### H1c — ценность нелинейности относительно linear predictive graph

```text
CatBoost-Directed-Dynamic < Linear-Directed-Dynamic
```

Три контраста образуют одно confirmatory family `directed_core3`; к их
двусторонним p-values применяется Holm correction.

### 2.3. Regime-гипотеза

#### H2 — состояние рынка модерирует ценность динамики графа

Пусть

```text
d[t] = daily QLIKE(CatBoost-Directed-Current)
       - daily QLIKE(CatBoost-Directed-Dynamic).
```

Проверяется модель

```text
d[t] = a + b * stress_z[t] + error[t].
```

Гипотеза:

```text
H2: b > 0.
```

То есть преимущество Dynamic должно возрастать при повышенном market stress.
H2 является отдельной заранее заданной гипотезой. Сообщаются односторонний и
двусторонний HAC p-value; основной направленный вывод использует
односторонний p-value.

### 2.4. Market-residual robustness

#### H3 — локальная сеть сохраняет сигнал после удаления market factor

Для idiosyncratic realized variance проверяется контраст:

```text
Residual-CatBoost-Directed-Current
    - Residual-CatBoost-Directed-Dynamic.
```

H3 является robustness-гипотезой, а не частью `directed_core3`. Если ветка
IMOEX не проходит заранее заданный coverage audit, H3 не заменяется другим
market proxy и помечается как `not estimable under protocol`.

---

## 3. Данные, временная шкала и единица наблюдения

### 3.1. Исходные данные

Используются существующие артефакты `data_v02`:

```text
data_v02/model_data/node_features_all82.parquet
data_v02/model_data/forecast_samples.parquet
data_v02/model_data/rv_all82.parquet
data_v02/universe_v03/dynamic_universe.csv
data_v02/universe_v03/intraday_download_secids.csv
data_v02/graphs/raw_pearson_edges.parquet
data_v02/model_audit/graph_quality_pearson.csv
```

Добавляются 10-минутные свечи индекса IMOEX, полученные из MOEX ISS тем же
causal pipeline, что и свечи акций.

### 3.2. Динамическая вселенная

- Базовая вселенная: существующий dynamic top-40 universe v03.
- Глобальный список допустимых узлов: существующие 82 SECID.
- На каждой дате прогноз оценивается только по SECID, входящим в зафиксированный
  universe и прошедшим одинаковые target/feature QC.
- Состав вселенной не интерполируется назад и не определяется по будущей
  ликвидности.

### 3.3. Сегменты и остановка 2022 года

- Существующие temporal segment identifiers сохраняются без изменений.
- Ни одно rolling window, permutation, lag или graph transition не может
  переходить через разрыв 2022 года.
- Circular wrap между сегментами запрещен.

### 3.4. Intraday quality control

- Сохраняется strict `52/52` session rule из v09.
- Даты, признанные неполными или невалидными в v09, повторно не включаются.
- Для IMOEX применяется тот же календарь и тот же session audit.
- Любое изменение правила числа bins требует версии v10.1 до просмотра
  модельных результатов.

### 3.5. Target

Основной target:

```text
y_rv[i,t] = RV[i,t+1]
y_logrv[i,t] = log(RV[i,t+1])
```

Модели могут обучаться на `y_logrv`, но основная оценка всегда выполняется по
QLIKE в RV-scale после train-only retransformation/calibration, идентичной v09.

Дополнительные метрики:

- RMSE по log-RV;
- MAE по log-RV;
- calibration ratio `mean(pred_RV) / mean(y_RV)`;
- доля дат, на которых правая модель имеет меньшую дневную QLIKE.

### 3.6. Development и confirmatory holdout

```text
development_end = 2026-08-10
confirmatory_start = первый валидный торговый день после 2026-08-10
```

Confirmatory holdout закрывается после накопления **120 валидных прогнозных
дат**. До этого запрещено:

- считать агрегированные holdout-метрики;
- строить cumulative loss plots;
- сравнивать варианты графа по holdout;
- менять признаки, окна, top-k, гиперпараметры или модели на основании
  отдельных holdout-ошибок.

Разрешено проверять только технические свойства: наличие файла, число строк,
уникальность ключей, finite predictions и causal timestamp audit. Значения
target и loss в промежуточном техническом отчете не выводятся.

Если к 2027-03-31 не накоплено 120 валидных дат, анализ выполняется на
доступных данных и явно маркируется как underpowered confirmatory analysis.

---

## 4. Market factor и residual-данные

### 4.1. Первичный market factor

Единственный основной market factor для residual-ветки — 10-минутная
доходность IMOEX:

```text
r_m[d,b] = log(P_IMOEX[d,b]) - log(P_IMOEX[d,b-1]),
```

где `d` — торговая дата, `b` — 10-минутный bin.

Рыночная realized variance:

```text
RV_m[d] = sum_b r_m[d,b]^2.
```

Market HAR:

```text
mkt_logrv_d[t] = log(RV_m[t])
mkt_logrv_w[t] = mean(log(RV_m[t-4:t]))
mkt_logrv_m[t] = mean(log(RV_m[t-21:t])).
```

### 4.2. Causal residualization

Для каждого актива `i` и дня `d` параметры оцениваются только на прошлых
intraday observations из предыдущих 60 валидных торговых дней:

```text
r_i[s,b] = alpha_i[d] + beta_i[d] * r_m[s,b] + epsilon_i[s,b],
s in previous 60 valid dates before d.
```

После этого параметры применяются к текущему дню `d`:

```text
epsilon_i[d,b]
    = r_i[d,b] - alpha_hat_i[d] - beta_hat_i[d] * r_m[d,b].
```

Idiosyncratic realized variance:

```text
IRV_i[d] = sum_b epsilon_i[d,b]^2.
```

Правила:

- окно beta: 60 предыдущих валидных дат;
- минимум для оценки: 40 валидных дат;
- intercept включается;
- коэффициенты не используют observations текущего дня при оценке;
- beta не winsorize и не shrink без amendment;
- неотрицательность IRV следует из определения;
- `log(IRV)` допускается только при `IRV > 0`.

### 4.3. Coverage gate для residual-ветки

Residual-ветка запускается только если:

- IMOEX имеет strict-valid coverage не менее 95% дат основной equity sample;
- не менее 90% основных `date x SECID` строк имеют causal IRV;
- нет случаев использования current/future returns при оценке beta.

Если gate не пройден, H3 пропускается. PCA, equal-weighted market и другие
замены могут исследоваться только в отдельной версии протокола и не выбираются
по качеству прогноза.

---

## 5. Общие causal predictor features

Для всех моделей в момент origin `t` доступны только значения с timestamp
`<= t`.

### 5.1. Own-HAR

```text
har_logrv_d[i,t]
har_logrv_w[i,t]
har_logrv_m[i,t]
```

с горизонтами 1, 5 и 22 торговых дня.

### 5.2. Market-HAR

```text
mkt_logrv_d[t]
mkt_logrv_w[t]
mkt_logrv_m[t]
```

### 5.3. Source groups для directed graph learner

Для каждого возможного источника `j != i` используется одна группа:

```text
Z_j[t] = {
    src_logrv_d[j,t],
    src_logrv_w[j,t],
    src_logrv_m[j,t]
}
```

Все три признака источника удаляются/переставляются совместно. Разделение
одного источника на три независимых ребра запрещено.

В residual-ветке `logRV` заменяется на `logIRV` и соответствующие 5/22-дневные
средние.

---

## 6. Временная схема построения directed graphs

### 6.1. Graph refit schedule

```text
graph_refit_block = 20 research dates
graph_train_window = 504 previous research dates
importance_window = 126 immediately preceding research dates
```

На начале блока `s`:

1. importance window заканчивается в `s-1`;
2. graph-train window расположен непосредственно перед importance window;
3. оба окна находятся внутри одного temporal segment;
4. граф `A_s` фиксируется для прогнозных дат блока `s ... s+19`;
5. ни одна observation прогнозного блока не используется при построении `A_s`.

Для построения графа требуется минимум:

- 400 target-valid строк в graph-train window;
- 80 target-valid строк в importance window;
- 80% coverage source group в importance window.

Если target-node не проходит gate, directed graph для него на этом блоке
считается invalid.

### 6.2. Почему граф фиксируется на 20 дней

Это отделяет изменение структурной predictive relation от ежедневного
изменения входов и совпадает с refit block существующего walk-forward дизайна.
Для честного сравнения Pearson benchmark также строится только на refit dates и
фиксируется внутри того же 20-дневного блока.

Daily local-SHAP graph не входит в confirmatory v10. Он может быть добавлен
только как exploratory visualization после завершения основных тестов.

---

## 7. CatBoost-directed graph

### 7.1. Target-specific models

Для каждого допустимого target-node `i` обучается отдельная модель:

```text
logRV[i,t+1]
    = f_i(OwnHAR[i,t], MarketHAR[t], Z_1[t], ..., Z_N[t]).
```

CatBoost применяется как nonlinear tabular learner. SECID не используется как
категориальный признак, поскольку модели target-specific.

### 7.2. Замороженная гиперпараметрическая сетка

На development sample выбирается **одна глобальная конфигурация** для всех
target-nodes и всех последующих refit-блоков:

```text
depth in {3, 5}
learning_rate in {0.03, 0.05}
iterations in {300, 600}
l2_leaf_reg in {3, 10}
loss_function = RMSE on log-RV
random_seed = 260818
thread_count = 1
allow_writing_files = False
```

Всего 16 конфигураций. Выбор выполняется по средней validation QLIKE после
train-only RV calibration на существующих development walk-forward folds.

Запрещено:

- выбирать отдельные параметры по SECID;
- расширять grid после просмотра confirmatory holdout;
- применять early stopping на importance window;
- заменять QLIKE на RMSE как критерий выбора конфигурации.

### 7.3. Определение веса ребра

Для источника `j` и target `i` сначала вычисляется исходная importance-window
loss:

```text
L_base[i] = QLIKE(y_i, f_i(X)).
```

Затем вся группа `Z_j` подвергается одной и той же circular block shift внутри
importance window. Используются 20 допустимых сдвигов, сгенерированных один раз
из seed `260819`; абсолютная величина каждого сдвига не меньше 20 дат. Сдвиг не
переходит через temporal segment.

```text
importance[j -> i]
    = mean_q(QLIKE(y_i, f_i(X with shifted Z_j[q])) - L_base[i]).
```

Raw edge weight:

```text
w_raw[j -> i] = max(0, importance[j -> i]).
```

Для каждого target `i`:

1. источники сортируются по `w_raw[j -> i]`;
2. сохраняются не более `k = 5` положительных incoming edges;
3. веса нормируются до суммы 1 по incoming edges;
4. если положительных weights нет, node/block помечается graph-invalid.

Таким образом, ребро измеряет дополнительный прошлый predictive content в
единицах validation QLIKE. Оно не интерпретируется как структурная причинность.

### 7.4. Почему не используется обычный feature_importance

`PredictionValuesChange`, split counts и mean absolute SHAP не являются
основным определением ребра, поскольку:

- они не измеряют изменение целевой QLIKE;
- correlated sources могут делить importance произвольным образом;
- local SHAP меняется вместе со значениями входов даже при фиксированной
  predictive relation.

SHAP сохраняется только для диагностического объяснения отдельных прогнозов и
не участвует в выборе top-k.

---

## 8. Linear-directed benchmark

Linear graph строится по тем же target, base features, source groups,
train/importance windows, refit dates, block shifts и top-k.

Модель:

```text
ElasticNet(logRV[i,t+1] ~ OwnHAR + MarketHAR + all source groups).
```

Все признаки стандартизуются только по graph-train window.

Замороженная grid:

```text
alpha in {1e-4, 1e-3, 1e-2, 1e-1, 1}
l1_ratio in {0.25, 0.50, 0.75}
max_iter = 10000
```

Выбирается одна глобальная конфигурация по development validation QLIKE.
Edge weights определяются тем же grouped shifted-source QLIKE degradation, а
не величиной отдельного коэффициента. Это делает сравнение Linear и CatBoost
сопоставимым и изолирует вклад нелинейности.

---

## 9. Pearson blocked benchmark

Pearson graph сохраняет frozen v09 определение зависимости и graph window 60,
но для method comparison пересчитывается только на общей refit date `s` и
фиксируется на следующие 20 research dates.

- `W_corr = 60` прошлых research dates;
- edge score: absolute Pearson correlation;
- `top-k = 5`;
- union/symmetrization следует существующему v09 builder;
- после симметризации weights нормируются для вычисления neighbor features;
- граф не использует данные forecast block.

Оригинальный daily Pearson Dynamic v09 сообщается отдельно как historical
reference, но не входит в `directed_core3`, поскольку имеет другую частоту
обновления.

dCor не входит в основные v10-модели: v09 не обнаружил отличия от Pearson.

---

## 10. Directed graph features

Пусть `A[j,i,s]` — нормированный вес ребра `j -> i` на refit block `s`.

### 10.1. Current features

Для каждого target-node `i`:

```text
incoming_nbr_logrv_d[i,t] = sum_j A[j,i,s] * logRV[j,t]
incoming_nbr_logrv_w[i,t] = sum_j A[j,i,s] * logRV_w[j,t]
incoming_nbr_logrv_m[i,t] = sum_j A[j,i,s] * logRV_m[j,t]
incoming_degree[i,s]       = number of positive A[j,i,s]
incoming_strength[i,s]     = sum_j w_raw[j -> i]
```

### 10.2. Dynamic features

Изменение сравнивает два последовательных refit-графа `s` и `s-1`:

```text
incoming_jaccard_distance[i,s]
delta_incoming_strength[i,s]
weighted_incoming_edge_change[i,s]
incoming_edge_turnover[i,s]
```

где

```text
weighted_incoming_edge_change[i,s]
    = sum_j |A[j,i,s] - A[j,i,s-1]|,

incoming_edge_turnover[i,s]
    = 1 - |N_in[i,s] intersect N_in[i,s-1]|
          / |N_in[i,s] union N_in[i,s-1]|.
```

Если оба множества пусты, transition invalid. Если узел появился впервые,
graph-change features отсутствуют; они не заполняются нулями.

### 10.3. Основной scalar graph-change для regime test

```text
g_change[i,t] = weighted_incoming_edge_change[i,s].
```

Другие graph-change features не используются для выбора stress definition.

---

## 11. Forecasting models

Все HAR-модели являются вложенными и обучаются на одном common paired sample.

### 11.1. Baselines

```text
M0 HAR-RV
    OwnHAR

M1 HAR-Market
    OwnHAR + MarketHAR
```

### 11.2. Pearson models

```text
M2 Pearson-Current-Blocked
    M1 + Pearson current graph features

M3 Pearson-Dynamic-Blocked
    M2 + Pearson graph-change features
```

### 11.3. Linear-directed models

```text
M4 Linear-Directed-Current
    M1 + linear-directed current features

M5 Linear-Directed-Dynamic
    M4 + linear-directed graph-change features
```

### 11.4. CatBoost-directed graph models

```text
M6 CatBoost-Directed-Current
    M1 + CatBoost-directed current features

M7 CatBoost-Directed-Dynamic
    M6 + CatBoost-directed graph-change features
```

M0-M7 используют тот же линейный HAR estimation pipeline v09: fixed effects,
train-only scaling, expanding walk-forward refits, alpha selection и QLIKE
calibration.

Замороженная ridge grid:

```text
alpha in {0, 0.01, 0.1, 1, 10, 100}
```

Alpha выбирается отдельно для каждого model_id только на development validation
folds и затем фиксируется.

### 11.5. Direct nonlinear baseline

```text
M8 Direct-CatBoost
    target-specific CatBoost on OwnHAR + MarketHAR + all source groups.
```

Для M8 используется глобальная выбранная CatBoost-конфигурация. На каждом
20-дневном forecasting block модель переобучается на всех доступных прошлых
development/observed данных и выдает прогнозы блока без дальнейшего tuning.

M8 нужен, чтобы различать два утверждения:

- nonlinear source information полезна для прогноза;
- преобразование этой информации в граф полезно как forecasting representation.

M8 является diagnostic benchmark и не входит в `directed_core3`.

---

## 12. Regime definition

Для каждой origin date `t`:

```text
stress_z[t]
    = (logRV_m[t] - median(logRV_m[t-252:t-1]))
      / MAD(logRV_m[t-252:t-1]).
```

Правила:

- требуется минимум 126 прошлых market dates;
- median и MAD не включают будущие даты;
- numerator использует текущий известный `RV_m[t]`;
- при `MAD = 0` regime observation invalid;
- `stress_z` фиксированно ограничивается интервалом `[-5, 5]`;
- rolling quantile не выбирается по forecast performance.

Для описательных таблиц:

```text
calm:   stress_z <= trailing 80th percentile
stress: stress_z > trailing 80th percentile
transition: stress_z[t] above threshold and stress_z[t-1] not above threshold
```

Эти категории используются только для visualization/descriptive effect sizes.
Основной H2 test использует непрерывный `stress_z`.

---

## 13. Placebo и falsification tests

### 13.1. Temporal alignment placebo

Для CatBoost-directed graph-change features используется их состояние на один
полный refit block раньше:

```text
directed_change_lag20[i,t] = directed_change[i,t-20 research dates].
```

Сравнение выполняется на отдельном common sample:

```text
Lag20 Dynamic vs Real Dynamic.
```

Будущие даты, wrap и переход через segment запрещены.

### 13.2. Source-identity shuffle

В каждом refit block source labels переставляются одной фиксированной
перестановкой seed `260820`, при этом:

- target labels сохраняются;
- incoming degree и multiset weights сохраняются;
- связь конкретного source с target разрушается;
- перестановка строится только среди доступных nodes блока.

Сравнение:

```text
Shuffled-Identity Dynamic vs Real CatBoost-Directed-Dynamic.
```

### 13.3. Frozen-graph placebo

Последний граф, доступный на начале forecast segment, повторяется без обновления
до конца segment. Он проверяет, важна ли эволюция структуры, а не только
направленная topology.

### 13.4. Direct-model falsification

Если M8 Direct-CatBoost лучше HAR-Market, но M7 не лучше M6 и Pearson Dynamic,
вывод формулируется так:

> nonlinear cross-asset information присутствует, но ее graph compression не
> дает дополнительной forecasting value.

Это не считается подтверждением directed graph hypothesis.

---

## 14. Walk-forward обучение и защита от leakage

### 14.1. Forecasting refit

- Forecast refit block: 20 trading dates.
- Training history: expanding внутри temporal segment.
- Validation/test block predictions строятся моделью, обученной только на
  датах до начала блока.
- Scaling, fixed effects, missing-value rules и RV calibration fit только на
  training portion текущего блока.

### 14.2. Three-clock rule

Для каждой forecast origin `t` одновременно проверяются три времени:

```text
max graph-train target date < min importance date
max importance date < graph refit forecast block start
max feature source date <= forecast origin t
target date > forecast origin t
```

Нарушение любого условия делает весь block invalid.

### 14.3. Common sample fairness

Каждый парный контраст использует только точное пересечение ключей:

```text
(TRADEDATE, SECID, y_rv)
```

для обеих моделей. Для каждого model pair сохраняется SHA-256 ключей и target,
число rows/dates/secids, duplicate count, nonpositive predictions и nonfinite
losses.

---

## 15. Статистический анализ

### 15.1. Единица inference

Primary loss сначала усредняется по доступным SECID внутри даты:

```text
L_model[t] = mean_i QLIKE(y[i,t], pred[i,t]).
```

Затем для пары моделей:

```text
d[t] = L_left[t] - L_right[t].
```

Положительное `d[t]` означает преимущество правой модели.

### 15.2. Frozen inference parameters

```text
HAC lags = 10
circular block bootstrap length = 20 dates
bootstrap repetitions = 10000
bootstrap seed = 260821
confidence level = 95%
```

Сообщаются:

- mean daily loss difference;
- HAC standard error, z и two-sided p-value;
- one-sided p-value для заранее направленных H1a/H2;
- circular block bootstrap confidence interval;
- centered bootstrap p-values;
- positive-date fraction;
- median difference;
- результаты по неперекрывающимся 20-дневным блокам.

### 15.3. Multiple testing

Family `directed_core3`:

```text
H1a CatBoost Current vs CatBoost Dynamic
H1b Pearson Dynamic vs CatBoost Dynamic
H1c Linear Dynamic vs CatBoost Dynamic
```

Для трех two-sided HAC и bootstrap p-values применяется Holm correction.

H2 рассматривается отдельно как одна prespecified interaction hypothesis.
Placebo и H3 сообщаются как отдельные robustness families и не используются для
переопределения основного результата.

### 15.4. Интерпретация

Сильное подтверждение H1 требует одновременно:

- правильного знака эффекта;
- Holm-adjusted `p < 0.05` хотя бы для H1a;
- bootstrap CI H1a, не включающего 0;
- преимущества минимум в 55% forecast dates;
- отсутствия аналогичного выигрыша у temporal/identity placebo.

Suggestive evidence:

- одинаковый знак на development и confirmatory holdout;
- nominal `p < 0.10`, но adjusted threshold не пройден;
- placebo имеет меньший эффект, но контраст с placebo незначим.

При остальных результатах H1 считается не подтвержденной.

---

## 16. Feasibility gates до просмотра forecasting outcomes

Перед вычислением любых model-comparison losses должны пройти проверки:

### 16.1. Directed graph audit

- finite edge weights: 100%;
- отсутствие self-loops: 100%;
- incoming normalized weight sum равна 1 с tolerance `1e-8`;
- не более 5 incoming edges;
- graph-valid не менее 90% forecast dates;
- median graph-covered active nodes не менее 35;
- source timestamp строго раньше forecast target;
- одинаковые refit dates для Pearson, Linear и CatBoost graphs.

### 16.2. Model audit

- duplicate keys: 0;
- nonpositive RV predictions: 0;
- nonfinite losses: 0;
- одинаковый target hash внутри каждого парного сравнения;
- alpha/hyperparameters совпадают с development selection artifact;
- test outcome не использован при selection.

Если feasibility gate не пройден, разрешено исправлять только реализационную
ошибку. Изменение научного параметра требует amendment v10.1 до просмотра
outcome metrics и полного повторения всех моделей.

---

## 17. Разрешенные и запрещенные изменения

### Разрешено без новой версии

- исправление пути к файлу;
- Parquet/CSV fallback без изменения строк;
- chunking, caching, resume и уменьшение `thread_count`;
- исправление явной программной ошибки с полным повтором affected pipeline;
- добавление технических audit columns.

### Требует v10.1 до просмотра outcomes

- изменение train/importance/refit windows;
- изменение `k`;
- изменение CatBoost/ElasticNet grids;
- замена grouped shifted importance на SHAP или split importance;
- изменение market factor;
- изменение stress definition;
- добавление или удаление основной модели/контраста;
- изменение sample QC или universe.

### Запрещено после просмотра confirmatory outcomes

- выбирать лучший graph method;
- менять знак или семейство гипотез;
- расширять hyperparameter grid;
- удалять неудобные даты/активы без ранее заданного QC;
- переименовывать exploratory result в confirmatory;
- делать новый holdout из части уже просмотренного периода.

---

## 18. Планируемые скрипты и артефакты

### 18.1. Скрипты

```text
download_imoex_10m_v01.py
build_market_factor_residuals_v01.py
build_directed_predictive_graphs_v01.py
build_directed_graph_features_v01.py
train_directed_har_benchmark_v01.py
analyze_directed_results_v01.py
```

### 18.2. Market-factor outputs

```text
data_v02/market_factor_v10/
├── imoex_10m.parquet
├── market_rv.parquet
├── asset_market_betas.parquet
├── idiosyncratic_rv.parquet
├── market_factor_coverage.csv
└── market_factor_design.json
```

### 18.3. Graph outputs

```text
data_v02/graphs_v10/
├── catboost_directed_edges.parquet
├── linear_directed_edges.parquet
├── pearson_blocked_edges.parquet
├── directed_graph_quality.csv
├── directed_importance_diagnostics.csv
├── graph_refit_dates.csv
└── graph_design.json
```

### 18.4. Feature outputs

```text
data_v02/model_data/
├── directed_graph_features.parquet
├── residual_directed_graph_features.parquet
└── directed_feature_audit.csv
```

### 18.5. Development benchmark

```text
data_v02/benchmark_v10_directed_development/
├── design.json
├── feature_sets.json
├── hyperparameter_selection.csv
├── sample_coverage.csv
├── fairness_audit.csv
├── predictions_val.parquet
├── predictions_development.parquet
├── metrics.csv
├── paired_inference_core.csv
├── paired_inference_regime.csv
├── paired_inference_placebo.csv
└── daily_loss_differentials.csv
```

### 18.6. Confirmatory outputs

До закрытия 120 valid dates разрешены только:

```text
data_v02/benchmark_v10_directed_confirmatory/
├── frozen_design.json
├── frozen_hyperparameters.json
├── technical_coverage_no_losses.csv
├── fairness_audit_no_targets.csv
└── predictions_sealed.parquet
```

После закрытия holdout создаются:

```text
├── model_metrics.csv
├── paired_inference_core.csv
├── regime_inference.csv
├── placebo_inference.csv
├── residual_robustness.csv
└── FINAL_DIRECTED_RESULTS_V10.md
```

---

## 19. Последовательность выполнения

1. Сохранить этот документ и его SHA-256.
2. Скачать и аудировать IMOEX 10-minute candles.
3. Построить causal market RV, betas и IRV.
4. Реализовать общий graph-refit calendar.
5. Реализовать Linear-directed graph.
6. Реализовать CatBoost-directed graph.
7. Провести graph-only feasibility audit без forecasting outcomes.
8. Построить одинаковые directed/blocked-Pearson features.
9. На development data выбрать только объявленные global hyperparameters.
10. Запустить M0-M8 на development sample и явно назвать результат exploratory.
11. Проверить temporal, identity и frozen-graph placebo.
12. Заморозить `frozen_hyperparameters.json` и хэши входов.
13. Начать prequential predictions на новом holdout после 2026-08-10.
14. Не открывать aggregate confirmatory losses до 120 valid dates.
15. Выполнить один финальный confirmatory analysis по разделу 15.

---

## 20. Возможные научные выводы

### Сценарий A: H1a и H1b подтверждены

Направленная nonlinear predictive network содержит дополнительную информацию,
а ее эволюция улучшает прогноз beyond current graph и Pearson dependence graph.

### Сценарий B: H1a подтверждена, H1b нет

История направленного графа полезна, но новое определение ребра не превосходит
более простой Pearson benchmark.

### Сценарий C: M8 выигрывает, H1a/H1b не подтверждены

Cross-asset nonlinear information полезна, но graph representation не дает
дополнительной forecasting value.

### Сценарий D: H2 подтверждена при слабом среднем H1

Граф полезен не постоянно, а state-dependently во время market stress. Это
допустимый результат, только если H2 сохраняется на новом holdout.

### Сценарий E: H3 подтверждена, raw H1 слаба

Общий market mode маскирует локальную сеть; предиктивная динамика проявляется в
idiosyncratic volatility.

### Сценарий F: все гипотезы не подтверждены

В проверенной MOEX sample динамика dependence/predictive graphs не дает
устойчивой информации beyond own and market HAR. Такой результат остается
публикуемым как строгий placebo-controlled negative finding.

---

## 21. Краткая формулировка будущей статьи

Рабочее название:

> **From dependence to directed predictability: Do evolving financial networks
> improve realized-volatility forecasts?**

Основной contribution:

> Работа отделяет ненаправленную совместную динамику цен от направленной
> нелинейной предсказуемости, сравнивает graph representation с direct nonlinear
> forecast и проверяет temporal alignment, source identity, market-mode removal
> и state dependence в едином causal walk-forward протоколе.

---

## 22. Freeze declaration

После начала вычисления development model-comparison metrics этот документ не
редактируется задним числом. Все изменения оформляются отдельным amendment с:

- датой;
- причиной;
- перечнем измененных параметров;
- указанием, были ли к моменту изменения просмотрены outcome metrics;
- обязательным новым filename/version.

Финальная confirmatory формулировка определяется этим протоколом, а не знаком
полученного результата.

