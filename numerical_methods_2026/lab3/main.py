import csv
import matplotlib.pyplot as plt

CSV_FILE = "temperature.csv"
MAX_DEGREE = 4
FUTURE_MONTHS = [25,26,27]

def read_csv(file_name):
    """Зчитування CSV у два масиви x і y"""
    x = []
    y = []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x.append(int(row['Month']))
            y.append(float(row['Temp']))
    return x, y

def form_matrix(x, m):
    """Формуємо матрицю нормальних рівнянь"""
    A = [[0.0 for _ in range(m+1)] for _ in range(m+1)]
    for i in range(m+1):
        for j in range(m+1):
            A[i][j] = sum(x[k]**(i+j) for k in range(len(x)))
    return A

def form_vector(x, y, m):
    """Формуємо вектор правої частини"""
    b = [0.0 for _ in range(m+1)]
    for i in range(m+1):
        b[i] = sum(y[k] * x[k]**i for k in range(len(x)))
    return b

def gauss_solve(A, b):
    """Розв'язання системи методом Гауса з вибором головного елемента по стовпцях"""
    n = len(b)
    # Прямий хід
    for k in range(n):
        max_row = max(range(k, n), key=lambda i: abs(A[i][k]))
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k+1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    # Зворотній хід
    x_sol = [0.0 for _ in range(n)]
    for i in reversed(range(n)):
        x_sol[i] = (b[i] - sum(A[i][j]*x_sol[j] for j in range(i+1, n))) / A[i][i]
    return x_sol

def polynomial(x_vals, coef):
    """Обчислення полінома у точках x_vals"""
    return [sum(coef[i]*xi**i for i in range(len(coef))) for xi in x_vals]

def variance(y_true, y_approx):
    """Середньоквадратична помилка"""
    n = len(y_true)
    return sum((y_true[i]-y_approx[i])**2 for i in range(n)) / n


x, y = read_csv(CSV_FILE)
n_points = len(x)

# 4.2 Пошук оптимального ступеня полінома
variances = []
coef_list = []

for m in range(1, MAX_DEGREE+1):
    A = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    coef = gauss_solve([row[:] for row in A], b_vec[:])
    y_approx = polynomial(x, coef)
    var = variance(y, y_approx)
    variances.append(var)
    coef_list.append(coef)

optimal_m = variances.index(min(variances)) + 1
coef_opt = coef_list[optimal_m-1]
y_approx_opt = polynomial(x, coef_opt)

print("Дисперсії для m=1..{}: {}".format(MAX_DEGREE, variances))
print("Оптимальний ступінь полінома:", optimal_m)

# 4.3 Прогноз на наступні місяці
y_future = polynomial(FUTURE_MONTHS, coef_opt)
print("Прогноз температури на місяці {}: {}".format(FUTURE_MONTHS, y_future))

# 4.4 Похибка апроксимації
error = [y[i]-y_approx_opt[i] for i in range(n_points)]

# 4.5 Побудова графіків
plt.figure(figsize=(10,5))
plt.plot(x, y, 'o', label='Фактичні дані')
plt.plot(x, y_approx_opt, '-', label='Апроксимаційний поліном')
plt.xlabel('Місяць')
plt.ylabel('Температура')
plt.title('Апроксимація температури методом МНК')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(x, error, 'r-o', label='Похибка апроксимації')
plt.xlabel('Місяць')
plt.ylabel('Похибка')
plt.title('Похибка апроксимації')
plt.grid(True)
plt.legend()
plt.show()

# 4.6 Графік дисперсії від ступеня полінома
plt.figure(figsize=(8,4))
plt.plot(range(1, MAX_DEGREE+1), variances, 'bo-')
plt.xlabel('Ступінь полінома')
plt.ylabel('Дисперсія')
plt.title('Дисперсія апроксимації залежно від ступеня полінома')
plt.grid(True)
plt.show()

