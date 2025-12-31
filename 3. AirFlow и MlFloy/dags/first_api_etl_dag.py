"""
Первый полный MLOps-пайплайн с Airflow и MLflow.
Загружает встроенный датасет, обучает модель классификации и логирует всё в MLflow.
"""
from datetime import datetime, timedelta
import json
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from airflow import DAG
from airflow.operators.python import PythonOperator

# Настройки по умолчанию для DAG
default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Определение DAG
with DAG(
    dag_id='first_mlops_pipeline',
    default_args=default_args,
    description='Полный MLOps-пайплайн: от данных до модели с MLflow',
    schedule_interval=timedelta(days=1),  # Запускать раз в день
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'classification', 'mlflow'],
) as dag:

    # ---------- TASK 1: Извлечение данных (Extract) ----------
    def extract_data(**context):
        """
        Загружает встроенный датасет digits и преобразует в pandas DataFrame.
        Передает данные через XCom.
        """
        print("Загрузка датасета digits для классификации...")
        data = load_digits()
        # Создаем DataFrame для удобства
        df = pd.DataFrame(data.data, columns=[f'pixel_{i}' for i in range(data.data.shape[1])])
        df['target'] = data.target
        # Для демонстрации упростим задачу до бинарной классификации (цифра 0 vs не 0)
        df['target_binary'] = (df['target'] == 0).astype(int)
        # Конвертируем в JSON для передачи через XCom
        data_json = df.to_json(orient='split')
        context['ti'].xcom_push(key='raw_data', value=data_json)
        print(f"Данные загружены. Размер: {df.shape}. Целевой класс '0': {df['target_binary'].sum()} примеров.")

    # ---------- TASK 2: Предобработка данных (Transform) ----------
    def transform_data(**context):
        """
        Разделяет данные на обучающую и тестовую выборки.
        """
        ti = context['ti']
        data_json = ti.xcom_pull(task_ids='extract_task', key='raw_data')
        df = pd.read_json(data_json, orient='split')
        # Разделяем признаки и целевую переменную
        feature_cols = [col for col in df.columns if col.startswith('pixel_')]
        X = df[feature_cols]
        y = df['target_binary']
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # Сохраняем разделенные данные в XCom
        context['ti'].xcom_push(key='X_train', value=X_train.to_json(orient='split'))
        context['ti'].xcom_push(key='X_test', value=X_test.to_json(orient='split'))
        context['ti'].xcom_push(key='y_train', value=y_train.to_json(orient='split'))
        context['ti'].xcom_push(key='y_test', value=y_test.to_json(orient='split'))
        print(f"Данные разделены. Train: {X_train.shape}, Test: {X_test.shape}")

    # ---------- TASK 3: Обучение модели с логированием в MLflow (Train & Log) ----------
    def train_and_log_model(**context):
        """
        Обучает модель RandomForest, логирует параметры, метрики и саму модель в MLflow.
        """
        ti = context['ti']
        # Получаем данные из предыдущего таска
        X_train = pd.read_json(ti.xcom_pull(task_ids='transform_task', key='X_train'), orient='split')
        y_train = pd.read_json(ti.xcom_pull(task_ids='transform_task', key='y_train'), orient='split')
        # Конфигурация MLflow (укажите ваш URI, если MLflow запущен отдельно)
        # Для Docker Compose из предыдущих шагов используйте 'http://mlflow-tracking-server:5000'
        mlflow.set_tracking_uri("http://mlflow-tracking-server:5000")  # ИЛИ "http://localhost:5000"
        mlflow.set_experiment("Airflow_ML_Pipeline_Demo")
        # Параметры модели (можно вынести в Variables Airflow)
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            print(f"MLflow Run ID: {run.info.run_id}")
            # 1. Логируем параметры
            mlflow.log_params(model_params)
            # 2. Обучаем модель
            model = RandomForestClassifier(**model_params)
            model.fit(X_train, y_train)
            # 3. Логируем модель как артефакт
            mlflow.sklearn.log_model(model, "random_forest_model")
            # 4. Рассчитываем и логируем метрики на обучающей выборке
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            mlflow.log_metric("train_accuracy", train_accuracy)
            # 5. Создаем и логируем текстовый артефакт с отчетом
            report = classification_report(y_train, y_train_pred, output_dict=True)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(report, f, indent=4)
                mlflow.log_artifact(f.name, "classification_report")
                os.unlink(f.name)  # Удаляем временный файл
            # Передаем run_id и параметры модели в XCom для следующего таска
            context['ti'].xcom_push(key='mlflow_run_id', value=run.info.run_id)
            context['ti'].xcom_push(key='model_params', value=model_params)
            print(f"Обучение завершено. Run ID: {run.info.run_id}, Accuracy на трейне: {train_accuracy:.3f}")

    # ---------- TASK 4: Оценка модели и создание артефактов (Evaluate) ----------
    def evaluate_model(**context):
        """
        Оценивает модель на тестовой выборке, логирует итоговые метрики и визуализации в MLflow.
        """
        ti = context['ti']
        # Получаем данные и run_id
        X_test = pd.read_json(ti.xcom_pull(task_ids='transform_task', key='X_test'), orient='split')
        y_test = pd.read_json(ti.xcom_pull(task_ids='transform_task', key='y_test'), orient='split')
        run_id = ti.xcom_pull(task_ids='train_task', key='mlflow_run_id')
        model_params = ti.xcom_pull(task_ids='train_task', key='model_params')
        # Подключаемся к тому же run в MLflow
        mlflow.set_tracking_uri("http://mlflow-tracking-server:5000")
        # Загружаем модель из MLflow
        model_uri = f"runs:/{run_id}/random_forest_model"
        model = mlflow.sklearn.load_model(model_uri)
        # Предсказания и метрики
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        # Логируем итоговую метрику в MLflow
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("accuracy_diff", test_accuracy - ti.xcom_pull(task_ids='train_task', key='train_accuracy'))
            # Создаем и логируем матрицу ошибок
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            # Сохраняем график во временный файл и логируем как артефакт
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                fig.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(tmp_file.name, "evaluation_plots")
                os.unlink(tmp_file.name)
            plt.close(fig)
        print(f"Оценка завершена. Accuracy на тесте: {test_accuracy:.3f}")
        # Сохраняем предсказания в XCom для возможного дальнейшего использования
        context['ti'].xcom_push(key='test_predictions', value=y_pred.tolist())
        context['ti'].xcom_push(key='final_test_accuracy', value=test_accuracy)

    # ---------- TASK 5: Деплой / Сохранение результатов (Optional) ----------
    def save_results(**context):
        """
        Сохраняет ключевые результаты выполнения пайплайна в локальный файл.
        Это может быть отправной точкой для развертывания модели.
        """
        ti = context['ti']
        run_id = ti.xcom_pull(task_ids='train_task', key='mlflow_run_id')
        test_accuracy = ti.xcom_pull(task_ids='evaluate_task', key='final_test_accuracy')
        results = {
            'mlflow_run_id': run_id,
            'test_accuracy': test_accuracy,
            'pipeline_execution_time': datetime.now().isoformat(),
            'model_uri': f"runs:/{run_id}/random_forest_model"
        }
        # Сохраняем в файл (в папку, смонтированную в контейнере)
        os.makedirs('./ml_pipeline_output', exist_ok=True)
        filename = f"./ml_pipeline_output/results_{run_id[:8]}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Результаты сохранены в файл: {filename}")

    # ---------- Определение задач в DAG ----------
    extract_task = PythonOperator(
        task_id='extract_task',
        python_callable=extract_data,
        provide_context=True,
    )
    transform_task = PythonOperator(
        task_id='transform_task',
        python_callable=transform_data,
        provide_context=True,
    )
    train_task = PythonOperator(
        task_id='train_task',
        python_callable=train_and_log_model,
        provide_context=True,
    )
    evaluate_task = PythonOperator(
        task_id='evaluate_task',
        python_callable=evaluate_model,
        provide_context=True,
    )
    save_task = PythonOperator(
        task_id='save_task',
        python_callable=save_results,
        provide_context=True,
    )

    # ---------- Определение порядка выполнения ----------
    extract_task >> transform_task >> train_task >> evaluate_task >> save_task