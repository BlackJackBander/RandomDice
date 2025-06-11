Код, который генерирует 110 бросков, использует первые 100 для обучения ARIMAX модели, а затем сравнивает прогноз на последние 10 бросков с реальными значениями:

```r
# Установка seed для воспроизводимости
set.seed(123)

# Параметры
n_train <- 100  # Данные для обучения
n_test <- 10    # Данные для тестирования
n_total <- n_train + n_test
n_dice <- 10    # Число кубиков

# 1. Генерация коррелированных кубиков
library(clusterGeneration)
library(MASS)

# Создаем корреляционную матрицу
cor_matrix <- genPositiveDefMat(n_dice, covMethod = "eigen")$Sigma
cor_matrix <- cov2cor(cor_matrix)

# Генерируем броски кубиков
dice_rolls <- floor(pnorm(mvrnorm(n_total, rep(0, n_dice), cor_matrix)) * 6) + 1

# 2. Random Walk для 10-го кубика с влиянием других
weights <- runif(n_dice - 1, -0.3, 0.3)
rw_10 <- numeric(n_total)
rw_10[1] <- dice_rolls[1, 10]

for (t in 2:n_total) {
  influence <- sum(weights * dice_rolls[t - 1, 1:9])
  rw_10[t] <- rw_10[t - 1] + influence + sample(-1:1, 1)
  rw_10[t] <- max(min(rw_10[t], 6), 1)
}

# 3. Разделение на обучающую и тестовую выборки
y_train <- rw_10[1:n_train]
y_test <- rw_10[(n_train+1):n_total]
xreg_train <- dice_rolls[1:n_train, 1:9, drop = FALSE]
xreg_test <- dice_rolls[(n_train+1):n_total, 1:9, drop = FALSE]

# 4. Подбор и прогноз ARIMAX
library(forecast)
fit <- auto.arima(y_train, xreg = xreg_train)
forecast_result <- forecast(fit, h = n_test, xreg = xreg_test)

# Ограничиваем прогнозные значения
forecast_mean <- pmin(pmax(as.numeric(forecast_result$mean), 1), 6)
forecast_lower <- pmin(pmax(as.numeric(forecast_result$lower[, 2]), 1), 6)
forecast_upper <- pmin(pmax(as.numeric(forecast_result$upper[, 2]), 1), 6)

# 5. Сравнение прогноза с реальными данными
comparison <- data.frame(
  Roll = (n_train+1):n_total,
  Actual = y_test,
  Forecast = forecast_mean,
  Lower = forecast_lower,
  Upper = forecast_upper,
  Error = forecast_mean - y_test
)

# Выводим таблицу сравнения
print(comparison)

# 6. Визуализация
library(ggplot2)
plot_data <- data.frame(
  Roll = 1:n_total,
  Value = rw_10,
  Type = c(rep("Обучающая выборка", n_train), rep("Тестовая выборка", n_test)),
  Forecast = c(rep(NA, n_train), forecast_mean),
  Lower = c(rep(NA, n_train), forecast_lower),
  Upper = c(rep(NA, n_train), forecast_upper)
)

ggplot(plot_data, aes(x = Roll)) +
  geom_line(aes(y = Value, color = Type), linewidth = 1) +
  geom_line(aes(y = Forecast), color = "red", linetype = "dashed", linewidth = 1.2) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = 0.2, fill = "blue") +
  geom_vline(xintercept = n_train + 0.5, linetype = "dashed", color = "gray") +
  labs(title = "Сравнение прогноза ARIMAX с реальными данными",
       subtitle = "Вертикальная линия - начало прогнозируемого периода",
       x = "Номер броска", y = "Значение кубика",
       color = "Тип данных") +
  ylim(1, 6) +
  theme_minimal() +
  scale_color_manual(values = c("Обучающая выборка" = "blue", "Тестовая выборка" = "darkgreen"))

# 7. Оценка точности прогноза
cat("\nСредняя абсолютная ошибка (MAE):", mean(abs(comparison$Error)), "\n")
cat("Среднеквадратичная ошибка (MSE):", mean(comparison$Error^2), "\n")
```

### Ключевые особенности решения:

1. **Реалистичное тестирование**:
   - Модель обучается на первых 100 бросках
   - Прогнозируются последние 10 бросков
   - Реальные значения сравниваются с прогнозом

2. **Полная метрика оценки**:
   - Таблица сравнения с ошибками прогноза
   - Расчет MAE и MSE для количественной оценки точности
   - Визуальное отображение доверительных интервалов

3. **Улучшенная визуализация**:
   - Четкое разделение обучающей и тестовой выборок
   - Вертикальная линия для обозначения начала прогноза
   - Разные цвета для типов данных

4. **Стабильность прогноза**:
   - Все значения ограничены диапазоном 1-6
   - Учет влияния других кубиков через регрессоры

Это решение позволяет наглядно оценить качество прогноза модели ARIMAX в условиях, приближенных к реальным, когда у нас есть только исторические данные и нужно предсказать будущие значения.
