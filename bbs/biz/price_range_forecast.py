# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.font_manager as fm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %%
plt.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False

# %%
# 데이터 불러오기
file_path = r"C:\Users\Admin\Desktop\dev\pyprj\bbs\csv\apartment_rent_data_price.csv"
df = pd.read_csv(file_path, sep=';', encoding='utf-8')
# df

# %%
# 필요한 컬럼만 선택
df = df[['시군구', '전월세구분', '보증금(만원)', '월세금(만원)', '계약년월']]
df

# %%
# '시군구'에서 '구' 정보만 추출
df['구'] = df['시군구'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else None)
df.dropna(subset=['구'], inplace=True)

# %%
# 보증금(만원)과 월세금(만원) 컬럼의 콤마(,) 제거 및 숫자형으로 변환
df['보증금(만원)'] = df['보증금(만원)'].str.replace(',', '', regex=True).fillna('0')
df['월세금(만원)'] = df['월세금(만원)'].str.replace(',', '', regex=True).fillna('0')

df['보증금(만원)'] = pd.to_numeric(df['보증금(만원)'])
df['월세금(만원)'] = pd.to_numeric(df['월세금(만원)'])

# %%
# 보증금과 월세금 합산하여 '총 전월세 금액' 계산 (월세는 120개월로 환산)
df['총 전월세 금액'] = df['보증금(만원)'] + df['월세금(만원)'] * 120

# %%
# 비정상적인 극단값(outlier) 제거
# 예: 20억(200000만원)을 초과하는 데이터는 제외
df = df[df['총 전월세 금액'] <= 200000]

# %%
# 금액대별 분류 (1억원 단위)
bins = list(range(0, int(df['총 전월세 금액'].max()) + 10000, 10000))
labels = [f'{i // 10000}억~{(i + 10000) // 10000}억' for i in range(0, int(df['총 전월세 금액'].max()) + 10000, 10000)[:-1]]
df['금액대'] = pd.cut(df['총 전월세 금액'], bins=bins, labels=labels, right=False)


# %%
# 전세와 월세 데이터 분리
df_jeonse = df[df['전월세구분'] == '전세'].copy()
df_wolse = df[df['전월세구분'] == '월세'].copy()

# %%
# 구별, 금액대별 수요량 (거래 건수) 계산
demand_jeonse = df_jeonse.groupby(['구', '금액대']).size().unstack(fill_value=0)
demand_wolse = df_wolse.groupby(['구', '금액대']).size().unstack(fill_value=0)

# %%
print("전세 수요량 (일부):\n", demand_jeonse.head())
print("\n월세 수요량 (일부):\n", demand_wolse.head())

# %%
plt.figure(figsize=(20, 10))

# 전세 히트맵
plt.subplot(1, 2, 1)
sns.heatmap(demand_jeonse, annot=True, fmt='d', cmap='YlGnBu')
plt.title('구별 전세 금액대별 수요량 히트맵')
plt.xlabel('금액대')
plt.ylabel('구')

# 월세 히트맵
plt.subplot(1, 2, 2)
sns.heatmap(demand_wolse, annot=True, fmt='d', cmap='OrRd')
plt.title('구별 월세 금액대별 수요량 히트맵')
plt.xlabel('금액대(보증금+월세 환산)')
plt.ylabel('구')

plt.tight_layout()
plt.show()

# %%
# '계약년월'을 날짜 형식으로 변환
df['계약년월'] = pd.to_datetime(df['계약년월'].astype(str), format='%Y%m')

# %%
# 금액대별 분류 (1억원 단위)
bins = list(range(0, int(df['총 전월세 금액'].max()) + 10000, 10000))
labels = [f'{i // 10000}억~{(i + 10000) // 10000}억' for i in range(0, int(df['총 전월세 금액'].max()) + 10000, 10000)[:-1]]
df['금액대'] = pd.cut(df['총 전월세 금액'], bins=bins, labels=labels, right=False)


# %%
# 데이터에 포함된 모든 구 리스트 확인
print("데이터에 포함된 구 리스트:", df['구'].unique())

# %%
# 구별, 금액대별 수요량 계산
demand_by_gu = df.groupby(['구', '금액대']).size().unstack(fill_value=0)

# %%
# 스케일링
scaler = StandardScaler()
scaled_demand = scaler.fit_transform(demand_by_gu)

# %%
# K-Means 모델 학습
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
demand_by_gu['cluster'] = kmeans.fit_predict(scaled_demand)
print(demand_by_gu['cluster'])

# %%
# PCA를 이용한 군집화 결과
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_demand)
print(reduced_data)

# %%
# 군집 시각화
plt.figure(figsize=(12, 10))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=demand_by_gu['cluster'], palette='viridis', s=200)
plt.title('구별 수요량 패턴 군집화 결과 (K-Means + PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
for i, gu in enumerate(demand_by_gu.index):
    plt.text(reduced_data[i, 0] + 0.05, reduced_data[i, 1], gu)
plt.show()

# %%
# 구별 월별 총 거래 건수 시계열 데이터 생성
time_series_data = df.groupby(['계약년월', '구']).size().reset_index(name='거래건수')
print(time_series_data)

# %%
# 각 구별로 예측 수행 (딕셔너리로 미리 변수잡기)
accuracy_results = {}
gu_list = time_series_data['구'].unique()

# %%
for gu in gu_list:
    gu_data = time_series_data[time_series_data['구'] == gu].rename(columns={'계약년월': 'ds', '거래건수': 'y'})
    
    # 데이터가 4개월(훈련 최소 1 + 테스트 3) 미만인 경우 건너뛰기
    if len(gu_data) < 4:
        continue

    # 데이터를 훈련 세트와 테스트 세트로 분할
    # 여기서는 마지막 6개월을 테스트 데이터로 사용
    train_data = gu_data[:-3]
    test_data = gu_data[-3:]

    # Prophet 모델 학습
    m = Prophet(seasonality_mode='multiplicative', daily_seasonality=False)
    m.fit(train_data)
    
    # 2025년 10월, 11월, 12월 예측 (3개월)
    future = m.make_future_dataframe(periods=3, freq='M')
    forecast = m.predict(future)

    # 테스트 기간의 예측값만 추출
    test_forecast = forecast['yhat'].tail(3)

# 예측값에 결측치(NaN)가 있거나 개수가 다르면 건너뛰기
    if test_forecast.isnull().values.any() or len(test_data) != len(test_forecast):
        continue

# %%
    # 실제 값과 예측 값 비교
    y_true = test_data['y'].values
    y_pred = test_forecast.values

    # 오차 지표 계산
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(mae)
    print(rmse)
# %%
    # MAPE는 직접 계산 (0으로 나누는 오류 방지)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
    print(mape)

# %%
# 결과 저장
    accuracy_results[gu] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# %%
# 정확도 결과 출력
print("--- 구별 예측 모델 정확도 ---")
for gu, metrics in accuracy_results.items():
    print(f"\n--- {gu} ---")
    print(f"MAE (평균 절대 오차): {metrics['MAE']:.2f}")
    print(f"RMSE (제곱근 평균 제곱 오차): {metrics['RMSE']:.2f}")
    print(f"MAPE (평균 절대 백분율 오차): {metrics['MAPE']:.2f}%")

# %%
forecast_results = {}
for gu in gu_list:
    gu_data = time_series_data[time_series_data['구'] == gu].rename(columns={'계약년월': 'ds', '거래건수': 'y'})
    
    if len(gu_data) > 3: # 예측에 충분한 데이터가 있는지 확인
        m = Prophet()
        m.fit(gu_data)
        future = m.make_future_dataframe(periods=3, freq='M')
        forecast = m.predict(future)
        forecast_results[gu] = forecast[['ds', 'yhat']].tail(3)
    else:
        forecast_results[gu] = pd.DataFrame(columns=['ds', 'yhat']) # 데이터 부족 시 빈 DF 저장

# %%
# 예측결과 시각화
plt.figure(figsize=(30, 20))
num_rows = int(np.ceil(len(gu_list) / 5))
num_cols = 5

for i, gu in enumerate(gu_list):
    ax = plt.subplot(num_rows, num_cols, i + 1)
    gu_data = time_series_data[time_series_data['구'] == gu].rename(columns={'계약년월': 'ds', '거래건수': 'y'})
    
    if not forecast_results[gu].empty:
        forecast_df = forecast_results[gu]
        bar_data = pd.concat([gu_data, forecast_df.rename(columns={'yhat': 'y'})], ignore_index=True)
        bar_data['ds_str'] = bar_data['ds'].dt.strftime('%Y-%m')
        
        bar_data['type'] = '실제'
        bar_data.loc[bar_data['ds'].isin(forecast_df['ds']), 'type'] = '예측'
        
        sns.barplot(x='ds_str', y='y', hue='type', dodge=False, ax=ax, data=bar_data)
        ax.set_title(f'{gu} 거래량 예측', fontsize=20)
        ax.set_xlabel('날짜', fontsize=12)
        ax.set_ylabel('거래 건수', fontsize=12)
        ax.legend(title='데이터 유형', loc='upper left', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
    else:
        ax.text(0.5, 0.5, '데이터 부족', ha='center', va='center', fontsize=25, color='red')
        ax.set_title(f'{gu} 거래량', fontsize=20)
        ax.set_xlabel('날짜', fontsize=12)
        ax.set_ylabel('거래 건수', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
plt.tight_layout()
plt.show()


# %%
#     # 막대 그래프용 데이터 프레임 생성
#     bar_data = gu_data.copy()
    
#     # 예측 데이터 추가
#     forecast_df = forecast_results[gu]
#     bar_data = pd.concat([bar_data, forecast_df.rename(columns={'yhat': 'y'})], ignore_index=True)
    
#     # 예측 데이터 구분용 컬럼 생성
#     bar_data['type'] = '실제'
#     bar_data.loc[bar_data['ds'].isin(forecast_df['ds']), 'type'] = '예측'
    
#     # 그래프 시각화 (막대 그래프)
#     ax = plt.subplot(i + 1)
#     sns.barplot(x=bar_data['ds'].dt.strftime('%Y-%m'), y=bar_data['y'], hue=bar_data['type'], dodge=False, ax=ax)

#     ax.set_title(f'{gu} 거래량 예측', fontsize=20)
#     ax.set_xlabel('날짜', fontsize=12)
#     ax.set_ylabel('거래 건수', fontsize=12)
#     ax.legend(title='데이터 유형', loc='upper left', fontsize=10)
#     ax.tick_params(axis='x', rotation=45, labelsize=10)

# plt.tight_layout()
# plt.show()

# %%
