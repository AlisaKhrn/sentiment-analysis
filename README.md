# Анализ тональности текста на русском языке

Проект представляет собой дообученную модель [`ai-forever/ruBert-base`](https://huggingface.co/ai-forever/ruBert-base) для решения задачи классификации тональности русскоязычных текстов. Финальная модель определяет три класса: негативный, нейтральный и позитивный.

Модель тестировалась на обработанной версии датасета **[rureviews](https://github.com/sismetanin/rureviews)**. Подробная информация об используемых данных находится в файле `README.md` в директории `data`.

## Сравнение производительности различных моделей по метрике F1-score:

| Модель | F1 Score |
|:---|:---:|
| **final_model** | **87.49** |
| XLM-RoBERTa-Large | 78.81 |
| LaBSE | 78.47 |
| XLM-RoBERTa-Base | 78.28 |
| EnRuDR-BERT | 77.95 |
| RuDR-BERT | 77.91 |
| Conversational RuBERT | 77.78 |
| MBART-50-Large | 77.52 |
| MBARTRuSumGazeta | 77.51 |
| SOTA | 77.44 |
| RuBERT | 77.41 |
| SBERT-Large | 77.41 |
| MBART-50-Large-Many-to-Many | 77.24 |
| SlavicBERT | 77.16 |

*Источник: [sentiment-analysis-in-russian](https://github.com/sismetanin/sentiment-analysis-in-russian)*

## Модель развернута с использованием Docker контейнера с простым веб-интерфейсом для тестирования. Для тестирования:

1. Скачайте образ из Docker Hub:
```bash
docker pull alisakhrn/sentiment-analysis-app:1.0.0
```
2. Запустите контейнер на порту 8080:
```bash
docker run -d -p 8080:8080 alisakhrn/sentiment-analysis-app:1.0.0
```
3. Откройте веб-интерфейс по адресу: http://localhost:8080
