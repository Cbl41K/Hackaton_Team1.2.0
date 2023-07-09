#Это либа для упрощения работы с графиками
import matplotlib.pyplot as plt
import seaborn as sns

def matrix_graphs(data):
    sns.pairplot(data, size=5)
    plt.show()


def matrix_of_dependency_graphs(data, cols_to_analyse):
    """
    Строит матрицу графиков зависимости между выбранными столбцами.

    Args:
        data (pandas.DataFrame): Данные для анализа.
        cols_to_analyse (list): Список столбцов для анализа.
    """
    sns.pairplot(data[cols_to_analyse], size=3)
    plt.show()


def matrix_correlation(cols_to_analyse):
    """
    Строит матрицу корреляции между выбранными столбцами и выводит тепловую карту.

    Args:
        cols_to_analyse (pandas.DataFrame): Столбцы для анализа.
    """
    correlation_matrix = cols_to_analyse.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

def histogram(data, column, bins=None):
    """
    Строит гистограмму значений выбранного столбца.

    Args:
        data (pandas.DataFrame): Данные для анализа.
        column (str): Название столбца.
        bins (int or sequence, optional): Количество интервалов (столбцов) гистограммы. По умолчанию None.
    """
    plt.hist(data[column], bins=bins)
    plt.title(column)
    plt.show()


def boxplot(data):
    sns.boxplot(data=data, orient="h")
    plt.show()