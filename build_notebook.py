#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nbformat as nbf
from textwrap import dedent

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(dedent("""
# Анализ 5V для сценария Fraud Detection

Выбран датасет [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) и бизнес-сценарий Fraud Detection: необходимо снизить количество мошеннических транзакций на 30% при ограничении задержки обработки до 50 мс и поддержке обработки late-arriving данных.
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
## Организация работы и воспроизводимость
- Для запуска ноутбука требуется предварительно скачать файл `creditcard.csv` из датасета Kaggle и разместить его в папке `data/`.
- В примерах используются библиотеки `pandas`, `numpy`, `pyarrow`, `fastavro`, `seaborn`, `matplotlib`, `great_expectations` и `scikit-learn`. При отсутствии их можно установить командой `pip install -r requirements.txt` (см. список в разделе импорта).
- Все расчёты выполняются на оригинальном датасете без семплирования, чтобы сохранить дисбаланс классов, критичный для fraud detection.
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
from pathlib import Path
import math
from io import BytesIO
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyarrow as pa
import pyarrow.parquet as pq
from fastavro import writer, parse_schema

import great_expectations as ge
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.validator.validator import Validator
from great_expectations.core.batch import Batch

from IPython.display import Markdown, display

plt.style.use('seaborn-v0_8')
sns.set_theme(style='whitegrid')

DATA_PATH = Path('data/creditcard.csv')
if not DATA_PATH.exists():
    raise FileNotFoundError('Файл data/creditcard.csv не найден. Скачайте датасет с Kaggle и поместите в папку data/.')

df = pd.read_csv(DATA_PATH)
df.head()
""")))

cells.append(nbf.v4.new_markdown_cell("""## 1. Volume (объём данных)"""))

cells.append(nbf.v4.new_code_cell(dedent("""
num_rows, num_cols = df.shape
csv_size_bytes = DATA_PATH.stat().st_size
avg_bytes_per_row = csv_size_bytes / num_rows

parquet_buffer = BytesIO()
pq.write_table(pa.Table.from_pandas(df), parquet_buffer, compression='snappy')
parquet_size_bytes = parquet_buffer.getbuffer().nbytes

avro_fields = []
for col, dtype in zip(df.columns, df.dtypes):
    if np.issubdtype(dtype, np.integer):
        avro_type = 'long'
    else:
        avro_type = 'double'
    avro_fields.append({'name': col, 'type': avro_type})

avro_schema = {
    'doc': 'Credit card transaction record',
    'name': 'CreditCardRecord',
    'namespace': 'fraud',
    'type': 'record',
    'fields': avro_fields,
}

avro_buffer = BytesIO()
writer(avro_buffer, parse_schema(avro_schema), df.to_dict('records'))
avro_size_bytes = avro_buffer.getbuffer().nbytes

storage_formats = pd.DataFrame([
    {
        'Формат': 'CSV',
        'Размер, МБ': csv_size_bytes / 1024**2,
        'Коэффициент сжатия к CSV': 1.0,
    },
    {
        'Формат': 'Parquet (Snappy)',
        'Размер, МБ': parquet_size_bytes / 1024**2,
        'Коэффициент сжатия к CSV': parquet_size_bytes / csv_size_bytes,
    },
    {
        'Формат': 'Avro',
        'Размер, МБ': avro_size_bytes / 1024**2,
        'Коэффициент сжатия к CSV': avro_size_bytes / csv_size_bytes,
    },
]).set_index('Формат')

volume_stats = {
    'csv_size_bytes': csv_size_bytes,
    'parquet_size_bytes': parquet_size_bytes,
    'avro_size_bytes': avro_size_bytes,
    'avg_bytes_per_row': avg_bytes_per_row,
}

storage_formats.round({'Размер, МБ': 2, 'Коэффициент сжатия к CSV': 3})
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
seconds_per_day = 24 * 3600
adjusted_time = df['Time'] - df['Time'].min()
day_index = np.floor(adjusted_time / seconds_per_day).astype(int)
daily_counts = day_index.value_counts().sort_index()

daily_volume_mb = daily_counts * volume_stats['avg_bytes_per_row'] / 1024**2

daily_growth = (
    pd.DataFrame({
        'day_index': daily_counts.index,
        'transactions': daily_counts.values,
        'daily_volume_mb': daily_volume_mb.values,
    })
    .sort_values('day_index')
    .reset_index(drop=True)
)

daily_growth['cumulative_volume_mb'] = daily_growth['daily_volume_mb'].cumsum()
start_timestamp = pd.Timestamp('2013-01-01')
daily_growth['date'] = start_timestamp + pd.to_timedelta(daily_growth['day_index'], unit='D')

avg_daily_mb = daily_growth['daily_volume_mb'].mean()
yearly_volume_gb = avg_daily_mb * 365 / 1024
months_to_tb = (1024 * 1024) / avg_daily_mb / (365 / 12)

volume_stats.update({
    'avg_daily_mb': avg_daily_mb,
    'yearly_volume_gb': yearly_volume_gb,
    'months_to_tb': months_to_tb,
})

daily_growth
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(data=daily_growth, x='date', y='cumulative_volume_mb', marker='o', ax=ax)
ax.set_title('Кумулятивный рост объёма данных')
ax.set_xlabel('Дата')
ax.set_ylabel('Объём, МБ')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
""")))

cells.append(nbf.v4.new_code_cell(dedent('''
summary_md = f"""
- **Размер исходного CSV:** {volume_stats['csv_size_bytes'] / 1024**2:.2f} МБ (~{num_rows} записей).
- **Средний объём данных за день:** {volume_stats['avg_daily_mb']:.2f} МБ, что даёт {volume_stats['yearly_volume_gb']:.2f} ГБ в год.
- **Порог 1 ТБ будет достигнут примерно через:** {volume_stats['months_to_tb']:.1f} месяцев при текущей скорости роста (~40 лет).
"""
display(Markdown(summary_md))
''')))

cells.append(nbf.v4.new_code_cell(dedent("""
cost_per_gb_month = 1.2  # RUB, тариф Yandex Object Storage Standard (предположение)
yearly_storage_cost_rub = volume_stats['yearly_volume_gb'] * cost_per_gb_month * 12

spark_threshold_gb = 5
spark_days = spark_threshold_gb * 1024 / volume_stats['avg_daily_mb']

cost_table = pd.DataFrame([
    {'Показатель': 'Годовой объём (Parquet)', 'Значение': f"{volume_stats['yearly_volume_gb']:.2f} ГБ"},
    {'Показатель': 'Стоимость хранения за год', 'Значение': f"{yearly_storage_cost_rub:,.0f} ₽"},
    {'Показатель': 'Дней до накопления ~5 ГБ', 'Значение': f"{spark_days:.0f} дней"},
])

volume_stats['yearly_storage_cost_rub'] = yearly_storage_cost_rub
volume_stats['spark_days'] = spark_days

cost_table
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
**Горячие vs холодные данные.** При среднем объёме ~72 МБ/сутки целесообразно хранить последние 30 дней (~2.1 ГБ) в оперативно доступном слое (например, в объектном хранилище + таблицы Delta Lake) для онлайн-моделей. Исторические данные старше 90 дней можно архивировать в холодное хранилище (Iceberg/Glacier) с более дешёвым тарифом, сохранив возможность периодического переобучения.

**Переход от Pandas к Spark.** Порог ~5 ГБ (~9.6 млн строк) будет достигнут через ~71 дней. Для обучения и инференса модели, учитывающей temporal features, уже на горизонте квартала рационально планировать миграцию на распределённую обработку (Spark/Flink), чтобы выдержать рост и расширение фич.
""")))

cells.append(nbf.v4.new_markdown_cell("""## 2. Velocity (скорость поступления данных)"""))

cells.append(nbf.v4.new_code_cell(dedent("""
base_time = pd.Timestamp('2013-01-01')
df_time = df.assign(event_time=base_time + pd.to_timedelta(df['Time'], unit='s'))

per_second = df_time.set_index('event_time').resample('1s').size()
full_range = pd.date_range(per_second.index.min(), per_second.index.max(), freq='1s')
per_second_full = per_second.reindex(full_range, fill_value=0)

per_second_stats = {
    'mean_rps': per_second_full.mean(),
    'std_rps': per_second_full.std(ddof=0),
    'cv': per_second_full.std(ddof=0) / per_second_full.mean(),
}

per_five_min = df_time.set_index('event_time').resample('5min').size()
per_hour = df_time.set_index('event_time').resample('1H').size()

velocity_table = pd.DataFrame([
    {'Метрика': 'Среднее событий/сек', 'Значение': f"{per_second_stats['mean_rps']:.2f}"},
    {'Метрика': 'Std событий/сек', 'Значение': f"{per_second_stats['std_rps']:.2f}"},
    {'Метрика': 'Коэффициент вариации', 'Значение': f"{per_second_stats['cv']:.2f}"},
    {'Метрика': 'Пиковая нагрузка (5 мин окно)', 'Значение': int(per_five_min.max())},
    {'Метрика': 'Среднее событий/час', 'Значение': f"{per_hour.mean():.0f}"},
    {'Метрика': 'Максимум событий/час', 'Значение': int(per_hour.max())},
])

velocity_stats = {
    'per_second_series': per_second_full,
    'per_five_min_series': per_five_min,
    'per_hour_series': per_hour,
    'peak_5min': int(per_five_min.max()),
}

velocity_table
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
fig, ax = plt.subplots(figsize=(9, 4))
per_hour_plot = per_hour.reset_index(name='transactions')
sns.lineplot(data=per_hour_plot, x='event_time', y='transactions', ax=ax)
ax.set_title('Распределение транзакций по часам')
ax.set_xlabel('Время')
ax.set_ylabel('Количество транзакций')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
batch_latency = 60 * 60  # сек
micro_batch_latency = 5 * 60
stream_latency = 2  # предполагаемый SLA на обработку true streaming

latency_table = pd.DataFrame([
    {'Режим обработки': 'Batch (каждый час)', 'Ожидаемая задержка, сек': batch_latency, 'Комментарий': 'Слишком поздно для онлайнового антифрода'},
    {'Режим обработки': 'Микробатчи (5 минут)', 'Ожидаемая задержка, сек': micro_batch_latency, 'Комментарий': 'Допустимо для мониторинга, но не выполняет SLA 50 мс'},
    {'Режим обработки': 'Стриминг (<50 мс)', 'Ожидаемая задержка, сек': stream_latency, 'Комментарий': 'Требует событийной архитектуры'},
])

latency_table
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
outage_minutes = 30
lost_records = per_second_stats['mean_rps'] * outage_minutes * 60
lost_volume_mb = lost_records * volume_stats['avg_bytes_per_row'] / 1024**2
lost_amount = lost_records * df['Amount'].mean()

velocity_stats.update({
    'lost_records_30min': lost_records,
    'lost_volume_mb_30min': lost_volume_mb,
    'lost_amount_30min': lost_amount,
})

loss_table = pd.DataFrame([
    {'Показатель': 'Потерянные записи за 30 мин сбоя', 'Значение': f"{lost_records:,.0f}"},
    {'Показатель': 'Объём потерянных данных', 'Значение': f"{lost_volume_mb:.2f} МБ"},
    {'Показатель': 'Потенциальный ущерб (средний чек)', 'Значение': f"{lost_amount:,.0f}"},
])

loss_table
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
limit_ms = 50
peak_rps = velocity_stats['peak_5min'] / (5 * 60)
service_capacity_per_worker = 1000 / limit_ms
headroom = 1.2  # 20% запас
required_workers = math.ceil((peak_rps / service_capacity_per_worker) * headroom)

resource_table = pd.DataFrame([
    {'Показатель': 'Пиковая нагрузка, событий/сек', 'Значение': f"{peak_rps:.2f}"},
    {'Показатель': 'Пропускная способность 1 воркера', 'Значение': f"{service_capacity_per_worker:.1f} событий/сек"},
    {'Показатель': 'Необходимое количество воркеров', 'Значение': required_workers},
])

velocity_stats['required_workers'] = required_workers

resource_table
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
**Вывод по Velocity.** При среднем потоке ~1.65 события/сек и пиках до 4.6 события/сек режим batch или микробатч приводит к задержке от минут до часа. Для выполнения SLA 50 мс необходим true streaming-процессинг (Flink/Spark Structured Streaming) с минимум двумя воркерами (один + резерв) и буферизацией поздних событий через Kafka. Сбой в 30 минут потенциально стоит ~262 тыс. у.е., поэтому критичен контроль отказоустойчивости и репликации.
""")))

cells.append(nbf.v4.new_markdown_cell("""## 3. Variety (разнообразие данных)"""))

cells.append(nbf.v4.new_code_cell(dedent("""
dtype_summary = pd.DataFrame({
    'Колонка': df.columns,
    'Тип': [str(t) for t in df.dtypes],
    'Уникальных значений': df.nunique().values,
    'Есть пропуски': df.isna().any().values,
})

dtype_summary
""")))

cells.append(nbf.v4.new_code_cell(dedent('''
example_record = df.iloc[0].to_dict()

variety_md = f"""
- Все {num_cols} полей — числовые (float/int), текстовых или полу-структурированных данных нет.
- Колонка `Time` хранит секунды с начала сбора без таймзоны; для feature engineering требуется перевод в календарные признаки.
- Колонки `V1`-`V28` — PCA-компоненты, название не несёт бизнес-смысла, важно сопровождать метаданными.
- Обнаружено {df.duplicated().sum()} дубликатов строк, что указывает на необходимость дедупликации до загрузки в витрины.

Пример записи:
```json
{json.dumps(example_record, indent=2)}
```
"""

display(Markdown(variety_md))
''')))

cells.append(nbf.v4.new_markdown_cell(dedent("""
**Работа с разнообразием.**
- Неструктурированных данных нет, поэтому дополнительные NLP/гео-обработки не требуются.
- Для временных паттернов нужно агрегировать по окнам (5 мин, час, сутки) и обогащать внешними источниками (часы, праздники, региональные особенности).
- Схема хранения: сырые данные в Parquet (bronze), нормализованные фичи в Delta Lake (silver), агрегаты и фичи для модели — в Feature Store (gold).
""")))

cells.append(nbf.v4.new_markdown_cell("""## 4. Veracity (достоверность данных)"""))

cells.append(nbf.v4.new_code_cell(dedent("""
quality_table = pd.DataFrame([
    {'Метрика качества': 'Пропуски по колонкам', 'Значение': f"{(df.isna().sum()==0).sum()} / {num_cols} колонок без пропусков"},
    {'Метрика качества': 'Отрицательные суммы', 'Значение': (df['Amount'] < 0).sum()},
    {'Метрика качества': 'Дубликаты строк', 'Значение': df.duplicated().sum()},
    {'Метрика качества': 'Доля мошенничества', 'Значение': f"{df['Class'].mean()*100:.4f}%"},
])

quality_table
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
ctx = ge.get_context()
engine = PandasExecutionEngine()
batch = Batch(data=df)
validator = Validator(execution_engine=engine, batches=[batch], data_context=ctx)

ge_results = {
    'No nulls in Time': validator.expect_column_values_to_not_be_null('Time'),
    'Amount non-negative': validator.expect_column_values_to_be_between('Amount', min_value=0),
    'Class is binary': validator.expect_column_values_to_be_in_set('Class', [0, 1]),
}

ge_summary = pd.DataFrame([
    {
        'Expectation': name,
        'Success': result['success'],
        'Unexpected %': result['result']['unexpected_percent'],
    }
    for name, result in ge_results.items()
])

ge_summary
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
**Воздействие проблем качества.** При ошибке меток даже на 1% (~2.8 тыс. транзакций) возможен рост FN и прямые потери >~0.6 млн у.е. Поэтому нужно:
- выполнять дедупликацию и контроль меток до поступления в real-time поток,
- ежедневно запускать data-quality проверки (Great Expectations + Airflow) и алерты на рост доли пропусков,
- логировать поздние события и пересчитывать фичи ретроспективно.
""")))

cells.append(nbf.v4.new_markdown_cell("""## 5. Value (ценность данных)"""))

cells.append(nbf.v4.new_code_cell(dedent("""
total_transactions = len(df)
total_fraud = int(df['Class'].sum())
fraud_amount_total = df.loc[df['Class'] == 1, 'Amount'].sum()
reduction_target = 0.30
eur_to_rub = 100  # предположение о курсе EUR/RUB
potential_savings_rub = fraud_amount_total * reduction_target * eur_to_rub

compute_cost_rub = 2 * 18 * 24 * 365  # два стриминг-воркера по 18 ₽/час
kafka_cost_rub = 6000 * 12  # управляемый кластер Kafka
storage_cost_rub = volume_stats['yearly_storage_cost_rub']
development_cost_rub = 2 * 220_000 * 3  # 2 инженера, 3 месяца

annual_cost_rub = compute_cost_rub + kafka_cost_rub + storage_cost_rub + development_cost_rub
roi = (potential_savings_rub - annual_cost_rub) / annual_cost_rub
payback_months = annual_cost_rub / (potential_savings_rub / 12)

value_table = pd.DataFrame([
    {'Показатель': 'Количество транзакций', 'Значение': f"{total_transactions:,}"},
    {'Показатель': 'Количество мошеннических транзакций', 'Значение': total_fraud},
    {'Показатель': 'Сумма мошенничества (EUR)', 'Значение': f"{fraud_amount_total:,.2f}"},
    {'Показатель': 'Цель (30%) — предотвращено, EUR', 'Значение': f"{fraud_amount_total * reduction_target:,.2f}"},
    {'Показатель': 'Потенциальная выгода (RUB)', 'Значение': f"{potential_savings_rub:,.0f}"},
    {'Показатель': 'Годовые затраты, RUB', 'Значение': f"{annual_cost_rub:,.0f}"},
    {'Показатель': 'ROI', 'Значение': f"{roi*100:.1f}%"},
    {'Показатель': 'Срок окупаемости', 'Значение': f"{payback_months:.1f} месяцев"},
])

value_stats = {
    'potential_savings_rub': potential_savings_rub,
    'annual_cost_rub': annual_cost_rub,
    'roi': roi,
    'payback_months': payback_months,
}

value_table
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
fraud_df = df.loc[df['Class'] == 1, ['Amount']].copy()
fraud_df = fraud_df.sort_values('Amount', ascending=False)
fraud_df['cum_amount'] = fraud_df['Amount'].cumsum()
fraud_df['fraction_records'] = (np.arange(len(fraud_df)) + 1) / len(fraud_df)
fraud_df['cum_amount_share'] = fraud_df['cum_amount'] / fraud_amount_total

fig, ax = plt.subplots(figsize=(7, 4))
sns.lineplot(data=fraud_df, x='fraction_records', y='cum_amount_share', ax=ax)
ax.set_title('Ценность vs объём обработанных данных')
ax.set_xlabel('Доля обработанных мошеннических транзакций')
ax.set_ylabel('Кумулятивная доля предотвращённого ущерба')
ax.axvline(0.2, color='red', linestyle='--', label='20% записей')
ax.axhline(fraud_df.loc[fraud_df['fraction_records'] <= 0.2, 'cum_amount_share'].max(), color='gray', linestyle=':')
ax.legend()
plt.tight_layout()
plt.show()
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
def categorize_abc(value):
    if value <= 0.8:
        return 'A'
    elif value <= 0.95:
        return 'B'
    return 'C'

fraud_df['abc_category'] = fraud_df['cum_amount_share'].apply(categorize_abc)
abc_summary = fraud_df.groupby('abc_category').agg(
    transactions=('Amount', 'count'),
    value_share=('Amount', lambda x: x.sum() / fraud_amount_total),
)
abc_summary['transactions_share'] = abc_summary['transactions'] / abc_summary['transactions'].sum()
abc_summary = abc_summary.rename(index={'A': 'A (top value)', 'B': 'B', 'C': 'C'}).reset_index()
abc_summary
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
**Мониторинг ценности.**
- Отслеживать precision/recall, FPR, долю предотвращённых сумм и факт возвратов каждые 15 минут.
- Пересчитывать ROI ежеквартально, учитывая реальный курс валют и обновлённые тарифы инфраструктуры.
- Через год при росте конкуренции и эволюции мошеннических схем ожидается повышение ценности данных: накопленные исторические последовательности позволят улучшить модели sequence modeling (LSTM/Transformers) и повысить recall на 3-5 п.п., что увеличит экономию.
""")))

cells.append(nbf.v4.new_markdown_cell("""## 6. Технологический стек и рекомендации"""))

cells.append(nbf.v4.new_code_cell(dedent("""
technology_table = pd.DataFrame([
    {
        'Слой': 'Хранилище сырых данных',
        'Рекомендуемая технология': 'Yandex Object Storage + Delta Lake',
        'Альтернативы': 'HDFS, S3, Iceberg',
        'Обоснование': 'Дешёвое хранение, версионирование, поддержка ACID для апдейтов поздних событий',
        'Риски': 'Стоимость запросов при горячем доступе, необходимость настроить версионирование',
    },
    {
        'Слой': 'Стриминг',
        'Рекомендуемая технология': 'Apache Kafka + Kafka Connect',
        'Альтернативы': 'Pulsar, RabbitMQ',
        'Обоснование': 'Гарантии доставки, re-play, поддержка late-arriving событий',
        'Риски': 'Требует DevOps-поддержки и мониторинга лагов',
    },
    {
        'Слой': 'Обработка',
        'Рекомендуемая технология': 'Apache Flink (или Spark Structured Streaming)',
        'Альтернативы': 'Kafka Streams, Storm',
        'Обоснование': 'Низкая задержка <50 мс, stateful стриминг, windows',
        'Риски': 'Сложность отладки stateful-джобов, необходимость чекпойнтов',
    },
    {
        'Слой': 'ML/Feature Store',
        'Рекомендуемая технология': 'Feast + MLflow',
        'Альтернативы': 'Hopsworks, Tecton',
        'Обоснование': 'Повторное использование фич онлайн/оффлайн, версионирование моделей',
        'Риски': 'Интеграция с онлайн-хранилищем (Redis/ClickHouse)',
    },
    {
        'Слой': 'Оркестрация и качество',
        'Рекомендуемая технология': 'Apache Airflow + Great Expectations',
        'Альтернативы': 'Prefect, Dagster',
        'Обоснование': 'Планирование DAG, запуск data-quality чеков, SLA мониторинг',
        'Риски': 'Нагрузка на scheduler, необходимость резервирования metadata DB',
    },
])

technology_table
""")))

cells.append(nbf.v4.new_markdown_cell(dedent(r"""
```mermaid
flowchart LR
    subgraph Ingestion
        A[Сырые транзакции] -->|REST/API| B[Kafka]
        B --> C[Kafka Connect]
    end
    subgraph Processing
        C --> D[Apache Flink Streaming]
        D -->|Features| E[Feature Store (Feast)]
        D -->|Alerts| F[Fraud API]
    end
    subgraph Storage
        D --> G[Delta Lake / Object Storage]
        E --> H[MLflow Model Registry]
        H --> F
    end
    subgraph Monitoring
        D --> I[Prometheus/Grafana]
        G --> J[Airflow + Great Expectations]
    end
```
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
roadmap = pd.DataFrame([
    {'Этап': '1. Подготовка (0-1 месяц)', 'Содержание': 'Настройка Kafka, Object Storage, загрузка исторических данных', 'Метрики успеха': 'Время загрузки < 10 мин, покрытие 100% данных'},
    {'Этап': '2. Стриминг-пайплайн (1-3 месяц)', 'Содержание': 'Разработка Flink job, реализация late data handling, чекпойнты', 'Метрики успеха': 'Задержка < 50 мс на 95-й перцентиль, падение <1% пакетов'},
    {'Этап': '3. ML и витрины (3-4 месяц)', 'Содержание': 'Фиче-стор, обучение модели, онлайн-инференс API', 'Метрики успеха': 'ROC AUC > 0.98, время инференса < 40 мс'},
    {'Этап': '4. Data Quality & мониторинг (4-5 месяц)', 'Содержание': 'Airflow DAG, Great Expectations, алерты', 'Метрики успеха': 'Покрытие проверками > 90%, MTTR < 15 мин'},
    {'Этап': '5. Эксплуатация и оптимизация (5-6 месяц)', 'Содержание': 'A/B тестирование, тюнинг параметров, отчёт по ROI', 'Метрики успеха': 'Снижение мошенничества ≥30%, ROI > 5%'},
])

roadmap
""")))

cells.append(nbf.v4.new_markdown_cell(dedent("""
**Критические риски и их минимизация.**
- *Late-arriving данные*: использовать watermark и reprocessing окон, хранить сырой поток в Kafka до 7 суток.
- *Дисбаланс классов*: применять stratified sampling, cost-sensitive loss, мониторить drift.
- *Регуляторные требования*: логирование решений модели, explainability (SHAP), хранение данных в соответствии с 152-ФЗ.
- *Отказы инфраструктуры*: multi-AZ развёртывание Kafka/Flink, автоматические рестарты, SLA-наблюдение через Prometheus.

**Итог.** Предложенное решение учитывает 5V: сжатие и горячее хранение (Volume), стриминг и отказоустойчивость (Velocity), управление схемой и фичами (Variety), регулярные проверки качества (Veracity) и измеримый экономический эффект с положительным ROI (Value).
""")))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.12'
    }
}

nbf.write(nb, 'Fraud_Detection_5V.ipynb')
