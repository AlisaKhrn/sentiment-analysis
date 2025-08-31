## Данные

#### Обучающая, валидационная и тестовая выборка состоит из двух наборов данных:

1. **[nyuuzyou/wb-feedbacks](https://huggingface.co/datasets/nyuuzyou/wb-feedbacks)** - датасет из библиотеки Hugging Face, содержащий 194М примеров, включающих следующие поля:
- `text` - текст отзыва,
- `productValuation` - оценка товара (число от 1 до 5).

2. **[rureviews](https://github.com/sismetanin/rureviews)** - датасет из репозитория на GitHub - sismetanin/rureviews
- `review` - текст отзыва,
- `sentiment` - метка тональности (positive, neautral, negative).

В дальнейшем по этому набору тестируется модель для сравнения с существующими решениями из работы на GitHub - [sismetanin
sentiment-analysis-in-russian](https://github.com/sismetanin/sentiment-analysis-in-russian)

#### Формирование выборок:
- **Тестовая выборка** полностью сформирована из набора `rureviews` и содержит 21k примеров.
- **Валидационная выборка** состоит из половины отзывов из набора `rureviews`, одинаково распределённых по классам и аналогично из набора `wb-feedbacks` и содержит также 21k примеров.
- **Тренировочная выборка** состоит из всех оставшихся отзывов из набора `rureviews` и дополнительно были взяты отзывы из `wb-feedbacks` таким образом, чтобы в конце концов из 168k отзывов по 56k примеров на каждый класс.
